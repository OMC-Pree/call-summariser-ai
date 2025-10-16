# summariser/process_summary/app.py
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Literal
import time
import base64
import requests

from utils import helper
from utils.s3_partitioner import get_s3_partitioner
import re
import boto3
from pydantic import BaseModel, ValidationError, Field
from botocore.exceptions import ClientError
from decimal import Decimal  # for DynamoDB numbers

# ----- Environment Variables -----
from constants import *
from prompts import (
    SUMMARY_PROMPT_TEMPLATE,
    CASE_CHECK_PROMPT_TEMPLATE,
    JSON_REPAIR_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_MESSAGE,
    CASE_CHECK_SYSTEM_MESSAGE,
    JSON_REPAIR_SYSTEM_MESSAGE
)

# AWS clients
s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
comprehend = boto3.client("comprehend")
dynamodb = boto3.resource("dynamodb")
ssm = boto3.client("ssm")
a2i = boto3.client("sagemaker-a2i-runtime", region_name=AWS_REGION)

JOB_TABLE_REF = dynamodb.Table(SUMMARY_JOB_TABLE)
 

# ---------- LLM response (summary) ----------
class ActionItem(BaseModel):
    description: str

class SentimentAnalysis(BaseModel):
    label: str
    confidence: float

class ThemeItem(BaseModel):
    id: str
    label: str
    group: Optional[str] = "General"  # Default to "General" if not provided
    confidence: float
    evidence_quote: Optional[str] = None

class ClaudeResponse(BaseModel):
    summary: str
    key_points: List[str]
    action_items: List[ActionItem]
    sentiment_analysis: SentimentAnalysis
    themes: List[ThemeItem] = Field(default_factory=list)

# ---------- Case check models ----------
Span = Tuple[int, int]
Status = Literal["Pass", "Fail", "NotApplicable", "Inconclusive"]

class CaseCheckResult(BaseModel):
    id: str
    status: Status
    confidence: float
    evidence_spans: List[Span] = Field(default_factory=list)  # avoid mutable default
    evidence_quote: Optional[str] = ""  # accept null; normalise to ""
    comment: Optional[str] = ""         # accept null; normalise to ""

class CaseCheckPayload(BaseModel):
    check_schema_version: str
    session_type: str
    checklist_version: str
    meeting_id: str
    model_version: str
    prompt_version: str
    results: List[CaseCheckResult]
    overall: dict

STARTER_SESSION_CHECKS = [
    # === Compliance Section (based on Excel) ===
    {
        "id": "call_recording_confirmed",
        "prompt": "Did the coach confirm call recording?",
        "required": True,
        "severity": "high"
    },
    {
        "id": "regulated_advice_given",
        "prompt": "Did the coach give regulated financial advice (e.g. specific product or investment recommendation)? "
                  "NOTE: This is not permitted. If the coach gave regulated advice, the status must be 'Fail'.",
        "required": True,
        "severity": "high"
    },
    {
        "id": "dob_confirmed",
        "prompt": "Was the client's date of birth confirmed?",
        "required": True,
        "severity": "medium"
    },
    {
        "id": "full_name_confirmed",
        "prompt": "Was the client's full name confirmed?",
        "required": True,
        "severity": "medium"
    },
    {
        "id": "financial_info_confirmed",
        "prompt": "Was the client's financial information confirmed (income, debts, savings, etc.)?",
        "required": True,
        "severity": "medium"
    },
    {
        "id": "partner_status_confirmed",
        "prompt": "Was the client's partner/marital status confirmed?",
        "required": False,
        "severity": "low"
    },
    {
        "id": "dependents_confirmed",
        "prompt": "Were dependents confirmed?",
        "required": False,
        "severity": "low"
    },
    {
        "id": "vulnerability_identified",
        "prompt": "Was any client vulnerability identified or discussed?",
        "required": True,
        "severity": "medium"
    },
    {
        "id": "citizenship_residency_confirmed",
        "prompt": "Were the client's citizenship and residency confirmed?",
        "required": True,
        "severity": "medium"
    },
    {
        "id": "fees_charges_explained",
        "prompt": "Were fees and charges correctly explained to the client?",
        "required": True,
        "severity": "medium"
    },
    {
        "id": "client_agreed_to_sign_up",
        "prompt": "Did the client agree to sign up to the service?",
        "required": False,
        "severity": "medium"
    },
    {
        "id": "pension_withdrawal_5y_discussed_50plus",
        "prompt": "If the client is 50 or older, did the coach discuss chances of withdrawing from their pension within 5 years?",
        "required": False,
        "severity": "high",
        "condition": "client_age>=50"
    },

    # === Macro Criteria Section (based on Excel) ===
    {
        "id": "holistic_explanation",
        "prompt": "Did the coach explain how they and Octopus Money help clients holistically (including getting advice)?",
        "required": True,
        "severity": "medium"
    },
    {
        "id": "action_ownership_explained",
        "prompt": "Did the coach explain that success depends on actions the client must complete?",
        "required": True,
        "severity": "medium"
    },
    {
        "id": "goal_understood_and_addressed",
        "prompt": "Did the coach fully understand the client's most important goal and give relevant suggestions?",
        "required": True,
        "severity": "medium"
    },
    {
        "id": "services_understood",
        "prompt": "Did the client clearly understand all relevant areas where Octopus Money can help (upfront and ongoing)?",
        "required": True,
        "severity": "medium"
    },
    {
        "id": "ecosystem_explained",
        "prompt": "Did the coach explain the Octopus Money ecosystem and the services already available to the client?",
        "required": False,
        "severity": "low"
    },
    {
        "id": "lifetime_journey_explained",
        "prompt": "Did the coach explain the financial pillars and the lifelong financial planning journey?",
        "required": False,
        "severity": "low"
    },
    {
        "id": "insurance_and_wills_actioned",
        "prompt": "Did the coach set insurance and wills actions where relevant (e.g., client has property or dependents)?",
        "required": False,
        "severity": "high",
        "condition": "has_property_or_dependents"
    },

    # === Suitability Section ===
    {
        "id": "suitability_for_ongoing_advice",
        "prompt": "Is the client suitable for ongoing advice?",
        "required": True,
        "severity": "high"
    }
]


# ---------- Helpers to make items DynamoDB-safe ----------
def _to_ddb_numbers(obj):
    """
    Recursively convert all Python floats in obj to Decimal for DynamoDB.
    Leaves ints/str/bool/None unchanged.
    """
    if isinstance(obj, float):
        # Convert via str to avoid binary float artifacts
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_ddb_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_ddb_numbers(v) for v in obj]
    return obj

# ---------- Status helpers ----------
def set_status(meeting_id: str, status: str, extra: dict | None = None, force: bool = False) -> None:
    status_up = status.upper()
    now_iso = datetime.now(timezone.utc).isoformat()
    exp_ts  = int((datetime.now(timezone.utc) + timedelta(days=90)).timestamp())

    # base SETs
    set_parts = ["#s = :s", "updatedAt = :u", "expiresAt = :e"]
    names = {"#s": "status"}
    vals  = {":s": status_up, ":u": now_iso, ":e": exp_ts}

    # add any extras as merged fields
    if extra:
        extra = _to_ddb_numbers(extra)
        for k, v in extra.items():
            names[f"#{k}"] = k
            vals[f":{k}"]  = v
            set_parts.append(f"#{k} = :{k}")

    update_expr = "SET " + ", ".join(set_parts)

    if status_up == "COMPLETED" or force:
        # Always allow COMPLETED or force to write without condition
        JOB_TABLE_REF.update_item(
            Key={"meetingId": meeting_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=vals,
        )
    else:
        # Don't clobber a previously-completed job
        vals[":done"] = "COMPLETED"  # Only add :done when needed for condition
        JOB_TABLE_REF.update_item(
            Key={"meetingId": meeting_id},
            UpdateExpression=update_expr,
            ConditionExpression="attribute_not_exists(#s) OR #s <> :done",
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=vals,
        )

def _extract_json_object(text: str) -> str:
    """
    Pull the first top-level {...} JSON object from a string that may
    contain prefaces like 'Here is the JSON output:' or code fences.
    Uses proper brace counting to handle nested structures correctly.
    """
    if not text:
        raise ValueError("Empty model output")
    t = text.strip()

    # Find the first opening brace
    start = t.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    # Use stack-based parsing to find the matching closing brace
    brace_count = 0
    in_string = False
    escaped = False

    for i in range(start, len(t)):
        char = t[i]

        if escaped:
            escaped = False
            continue

        if char == '\\' and in_string:
            escaped = True
            continue

        if char == '"' and not escaped:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    return t[start:i+1]

    # If we get here, the JSON is incomplete (no matching closing brace)
    # Check if what we have looks like it might be parseable as JSON
    incomplete_text = t[start:]

    # If the incomplete text is too short or doesn't look like JSON,
    # it's probably just a fragment - return the original text instead
    if len(incomplete_text) < 20 or not incomplete_text.strip().startswith('{'):
        return t

    return incomplete_text

def job_is_completed(meeting_id: str) -> bool:
    resp = JOB_TABLE_REF.get_item(Key={"meetingId": meeting_id})
    item = resp.get("Item") or {}

    # Not completed? definitely not done.
    if (item.get("status") or "").upper() != "COMPLETED":
        return False

    # Version-aware freshness check - only regenerate if content-affecting versions changed
    current = {
        "modelVersion":  MODEL_VERSION,
        "promptVersion": PROMPT_VERSION,
    }
    previous = {
        "modelVersion":  item.get("modelVersion"),
        "promptVersion": item.get("promptVersion"),
    }

    # Only consider stale if model or prompt version changed (these affect content)
    # Schema version and prefix changes don't require regeneration - just reformatting
    content_affecting_change = any(current[k] != previous.get(k) for k in current)

    # However, if the schema version is significantly different (major version change), regenerate
    current_schema = SCHEMA_VERSION
    previous_schema = item.get("schemaVersion", "1.0")

    # Extract major version numbers for comparison (e.g., "1.2" -> 1)
    try:
        current_major = int(current_schema.split('.')[0])
        previous_major = int(previous_schema.split('.')[0])
        major_schema_change = current_major != previous_major
    except (ValueError, IndexError):
        major_schema_change = current_schema != previous_schema

    is_stale = content_affecting_change or major_schema_change
    return not is_stale


# ----------- Zoom helpers -----------------
def _ssm_get(name, decrypt=False):
    return ssm.get_parameter(
        Name=f"{ZOOM_PARAM_PREFIX}/{name}", WithDecryption=decrypt
    )["Parameter"]["Value"]

def _ssm_put(name, value, secure=False):
    kwargs = dict(
        Name=f"{ZOOM_PARAM_PREFIX}/{name}", Value=value, Overwrite=True
    )
    kwargs["Type"] = "SecureString" if secure else "String"
    ssm.put_parameter(**kwargs)

def _get_zoom_token_from_ssm():
    try:
        tok = _ssm_get("access_token", True)
        exp = int(_ssm_get("access_token_expires_at"))
        if time.time() < exp - 60:
            return tok
    except Exception:
        pass
    acct = _ssm_get("account_id")
    cid = _ssm_get("client_id", True)
    sec = _ssm_get("client_secret", True)
    basic = base64.b64encode(f"{cid}:{sec}".encode()).decode()
    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={acct}"
    r = requests.post(
        url,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    tok = data["access_token"]
    exp = int(time.time()) + int(data.get("expires_in", 3600))
    _ssm_put("access_token", tok, True)
    _ssm_put("access_token_expires_at", str(exp), False)
    return tok


# --- Add near top (imports already include re, time, os, json) ---

VTT_TS = re.compile(r'^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}$')

def vtt_to_text(vtt: str) -> str:
    """
    Strip WEBVTT headers, cue numbers, timestamps and collapse blanks.
    Keeps 'Name:' speaker lines if present.
    """
    out = []
    for line in vtt.splitlines():
        s = line.strip()
        if not s:
            out.append("")  # keep paragraph separation
            continue
        if s.upper().startswith("WEBVTT"):
            continue
        if VTT_TS.match(s):
            continue
        if s.isdigit():  # cue number
            continue
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def _s3_put_text(bucket: str, key: str, text: str, content_type="text/plain"):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)


def pii_entities_chunked(text: str, chunk=4500, overlap=200, min_score=0.7, mask_types=None) -> list:
    """
    Run Comprehend PII detection over long text by sliding window.
    Returns deduped entities with offsets rebased to original text.
    """
    entities = []
    i, n = 0, len(text)

    while i < n:
        part = text[i:i+chunk]
        resp = comprehend.detect_pii_entities(Text=part, LanguageCode="en")
        for e in resp.get("Entities", []):
            if e["Score"] < min_score:
                continue
            if mask_types and e["Type"] not in mask_types:
                continue
            entity = dict(e)
            entity["BeginOffset"] += i
            entity["EndOffset"] += i
            entities.append(entity)
        i += max(1, chunk - overlap)

    # dedupe by offset+type
    seen, deduped = set(), []
    for e in entities:
        key = (e["BeginOffset"], e["EndOffset"], e["Type"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    return deduped

def normalise_roles(transcript: str, coach_name: str) -> str:
    """
    Replace speaker names in transcript with COACH or CLIENT.
    - Matches the coach_name (case-insensitive) and replaces with 'COACH'.
    - All other speakers are replaced with 'CLIENT'.
    - Works with Zoom VTT style: 'Name:' at the start of a line.
    """
    if not transcript or not coach_name:
        return transcript

    coach_pattern = re.compile(rf"^{re.escape(coach_name)}\s*:", re.IGNORECASE)
    speaker_pattern = re.compile(r"^([^:]{2,30}):")  # any 'Name:' up to 30 chars

    out_lines = []
    for line in transcript.splitlines():
        # if line starts with coach name
        if coach_pattern.match(line):
            out_lines.append(coach_pattern.sub("COACH:", line, count=1))
        elif speaker_pattern.match(line):
            out_lines.append(speaker_pattern.sub("CLIENT:", line, count=1))
        else:
            out_lines.append(line)
    return "\n".join(out_lines)

def _retry_http(fn, tries=4, base=0.5):
    for i in range(tries):
        try:
            return fn()
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", 0)
            if code in (429, 500, 502, 503, 504):
                time.sleep(base * (2 ** i))
                continue
            raise

def fetch_transcript_by_zoom_meeting_id(zoom_meeting_id: str) -> Tuple[str, str]:
    """
    Returns (plain_text_transcript, raw_vtt).
    """
    token = _get_zoom_token_from_ssm()

    def _list_files():
        url = f"https://api.zoom.us/v2/meetings/{zoom_meeting_id}/recordings"
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=15)
        if r.status_code == 404:
            raise RuntimeError("Zoom recordings not found yet (still processing).")
        r.raise_for_status()
        return r.json().get("recording_files", []) or []

    files = _retry_http(_list_files)

    vtt_url = None
    for f in files:
        if f.get("file_type") in ("TRANSCRIPT", "CC"):
            vtt_url = f.get("download_url")
            break
    if not vtt_url:
        raise RuntimeError("No transcript file (TRANSCRIPT/CC) available for this meeting.")

    def _download_vtt():
        r = requests.get(vtt_url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
        if r.status_code == 401:
            r = requests.get(f"{vtt_url}?access_token={token}", timeout=20)
        r.raise_for_status()
        return r.text

    raw_vtt = _retry_http(_download_vtt)
    plain = vtt_to_text(raw_vtt)

    # Optional S3 persistence (set SAVE_TRANSCRIPTS=true)
    if SAVE_TRANSCRIPTS:
        now = datetime.now(timezone.utc)
        partitioner = get_s3_partitioner()

        if ATHENA_PARTITIONED:
            # Save supplementary files (transcripts, VTT) to supplementary/ subdirectory
            base = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={now.year}/month={now.month:02d}/meeting_id={zoom_meeting_id}"
        else:
            # Legacy format for backward compatibility
            base = f"{S3_PREFIX}/{now:%Y}/{now:%m}/{zoom_meeting_id}"

        _s3_put_text(SUMMARY_BUCKET, f"{base}/zoom_raw.vtt", raw_vtt, "text/vtt")
        _s3_put_text(SUMMARY_BUCKET, f"{base}/transcript.txt", plain, "text/plain")

    return plain, raw_vtt

# ---------- Utilities ----------
def redact_text(text: str, entities: list) -> str:
    # Redact by character replacement to preserve offsets
    redacted = list(text)
    # Replace from the end to avoid offset shifts
    for entity in sorted(entities, key=lambda x: x["BeginOffset"], reverse=True):
        redacted[entity["BeginOffset"] : entity["EndOffset"]] = "[REDACTED]"
    return "".join(redacted)

def build_prompt(transcript: str) -> str:
    # Use replace instead of format to avoid issues with curly braces in transcript
    return SUMMARY_PROMPT_TEMPLATE.replace("{transcript}", transcript)


# ---------- Quality score calculation ----------
def _calculate_quality_score(data: ClaudeResponse, actions: list, themes: list) -> float:
    """
    Calculate a quality score (0.0-1.0) based on multiple factors:
    - Summary length and content quality
    - Number of actionable items
    - Theme identification
    - Sentiment confidence
    """
    score = 0.0

    # Summary quality (30% weight) - based on length and presence
    if data.summary and len(data.summary.strip()) > 50:
        summary_score = min(1.0, len(data.summary.strip()) / 500.0)  # normalize to 500 chars
        score += summary_score * 0.3

    # Action items quality (25% weight)
    if actions:
        action_score = min(1.0, len(actions) / 5.0)  # normalize to 5 actions
        score += action_score * 0.25

    # Theme identification (20% weight)
    if themes:
        theme_score = min(1.0, len(themes) / 4.0)  # normalize to 4 themes
        # Bonus for high confidence themes
        avg_confidence = sum(t.get("confidence", 0) for t in themes) / len(themes)
        theme_score += avg_confidence * 0.1
        score += min(1.0, theme_score) * 0.2

    # Key points quality (15% weight)
    if data.key_points:
        points_score = min(1.0, len(data.key_points) / 5.0)  # normalize to 5 points
        score += points_score * 0.15

    # Sentiment analysis presence (10% weight)
    if data.sentiment_analysis and data.sentiment_analysis.label and data.sentiment_analysis.label.strip():
        score += 0.1

    return min(1.0, max(0.0, score))  # clamp to [0.0, 1.0]

# ---------- Summary payload + saver ----------
def _build_summary_payload(meeting_id: str, coach_name:str, employer_name:str, source: str, data: ClaudeResponse) -> dict:
    """Schema 1.1 envelope + decision-ready metadata, driven by env versions."""
    now_iso = datetime.now(timezone.utc).isoformat() + "Z"
    sentiment_label = (
        data.sentiment_analysis.label
        if data.sentiment_analysis.label in ("Positive", "Neutral", "Negative")
        else "Neutral"
    )
    sentiment_confidence = max(0.0, min(1.0, float(data.sentiment_analysis.confidence)))
    actions = [
        {
            "id": f"A{i+1}",
            "text": a.description,
        }
        for i, a in enumerate(data.action_items or [])
    ]
    themes = [
        {
            "id": t.id,
            "label": t.label,
            "group": t.group or "General",  # Ensure group is never None
            "confidence": max(0.0, min(1.0, float(t.confidence))),  # Clamp to [0,1]
            "evidence_quote": t.evidence_quote or ""
        }
        for t in (data.themes or [])
    ]

    # Calculate quality score based on multiple factors
    quality_score = _calculate_quality_score(data, actions, themes)

    return {
        "summary_schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "meeting": {
            "id": meeting_id,
            "employerName": employer_name, #can be added later
            "coach": coach_name,
            "createdAt": now_iso,
        },
        "sentiment": {
            "label": sentiment_label,
            "confidence": sentiment_confidence,
            "evidence_spans": [],
        },
        "themes": themes,
        "summary": data.summary,
        "actions": actions,
        "call_metadata": {
            "source": source,
            "saved_at": now_iso,
            "insights_version": INSIGHTS_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model_version": MODEL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "prefix_version": S3_PREFIX,
        },
        "insights": {
            "action_count": len(actions),
            "theme_count": len(themes),
            "sentiment_label": sentiment_label,
            "is_escalation_candidate": (sentiment_label == "Negative"),
            "risk_flags": [],
            "categories": [],
        },
        "quality_score": quality_score,
    }

def _save_summary_json(meeting_id: str, payload: dict) -> tuple[str, str]:
    """
    Persist the summary JSON to a run-specific key and update a stable 'latest' key.
    Returns (run_key, latest_key).
    Only creates new files if summary.json doesn't already exist.
    """
    run_key, latest_key = _summary_object_keys(meeting_id)

    # Check if summary.json already exists
    try:
        s3.head_object(Bucket=SUMMARY_BUCKET, Key=latest_key)
        # File exists - return existing paths without creating duplicates
        print(f"Summary already exists for meeting {meeting_id}, skipping duplicate creation")
        return run_key, latest_key
    except Exception:
        # File doesn't exist - create new summary
        pass

    body = json.dumps(payload).encode("utf-8")

    # Only create the latest_key (summary.json), skip timestamped duplicates
    s3.put_object(
        Bucket=SUMMARY_BUCKET,
        Key=latest_key,
        Body=body,
        ContentType="application/json",
        # ServerSideEncryption="aws:kms",
        CacheControl="no-store",
    )

    # Return latest_key as both run_key and latest_key to avoid confusion
    return latest_key, latest_key


# ---------- Case check (no RAG) ----------
def _build_case_prompt(meeting_id: str, cleaned_transcript: str, checks: list) -> str:
    checklist_json = json.dumps(
        {"session_type": "starter_session", "version": "1", "checks": checks},
        ensure_ascii=False,
    )
    return CASE_CHECK_PROMPT_TEMPLATE.format(
        meeting_id=meeting_id,
        model_version=MODEL_VERSION,
        prompt_version=PROMPT_VERSION,
        checklist_json=checklist_json,
        cleaned_transcript=cleaned_transcript
    )

def _save_case_json(meeting_id: str, payload: dict, year: int = None, month: int = None) -> str:
    # Use provided year/month (from summary partition) or fallback to current date
    if year is None or month is None:
        now = datetime.now(timezone.utc)
        year = now.year
        month = now.month

    if ATHENA_PARTITIONED:
        # Save case check to supplementary/ subdirectory
        # Use the same year/month partition as the summary for consistency
        key = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={year}/month={month:02d}/meeting_id={meeting_id}/case_check.v{CASE_CHECK_SCHEMA_VERSION}.json"
    else:
        # Legacy format for backward compatibility
        key = f"{S3_PREFIX}/{year:04d}/{month:02d}/{meeting_id}/case_check.v{CASE_CHECK_SCHEMA_VERSION}.json"

    s3.put_object(
        Bucket=SUMMARY_BUCKET,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )
    helper.log_json("INFO", "CASE_CHECK_SAVED", meetingId=meeting_id, s3Key=key)
    return key

# ==== A2I additions START ====
def start_case_review_if_needed(meeting_id: str, case_payload: dict, redacted_transcript: str) -> Optional[str]:
    """
    Decide whether to trigger A2I. Returns HumanLoopName if started, else None.
    Triggers when overall pass_rate < 0.50 or if there are any 'high' severity fails.
    (You can tweak this later.)
    """
    try:
        if not A2I_FLOW_ARN_CASE:
            return None  # disabled unless env var set

        overall = case_payload.get("overall", {}) or {}
        pass_rate = float(overall.get("pass_rate") or 0.0)
        trigger = pass_rate < 0.50

        # Optional: scan for high-severity failures if present in results/comments
        if not trigger:
            severity_by_id = {c["id"]: c.get("severity", "low") for c in STARTER_SESSION_CHECKS}
            for r in case_payload.get("results", []) or []:
                if r.get("status") == "Fail" and severity_by_id.get(r.get("id")) == "high":
                    trigger = True
                    break

        if not trigger:
            return None

        # Keep payload small; reviewers can click through to full JSON later if needed.
        loop_name = _safe_loop_name("case", meeting_id)
        input_content = {
            "meeting_id": meeting_id,
            "pass_rate": pass_rate,
            "transcript_excerpt": (redacted_transcript or "")[:4000],
            "case_json": json.dumps(case_payload)[:4000]
        }

        a2i.start_human_loop(
            HumanLoopName=loop_name,
            FlowDefinitionArn=A2I_FLOW_ARN_CASE,
            HumanLoopInput={"InputContent": json.dumps(input_content)},
            DataAttributes={"ContentClassifiers": ["FreeOfPersonallyIdentifiableInformation"]}
        )
        return loop_name
    except ClientError as e:
        print(f"[A2I] start_human_loop failed: {e}")
    except Exception as e:
        print(f"[A2I] Unexpected error starting human loop: {e}")
    return None

def _safe_loop_name(prefix: str, meeting_id: str) -> str:
    """
    Produce a SageMaker A2I-safe loop name (alnum + hyphen, <= 63 chars).
    """
    base = re.sub(r'[^a-zA-Z0-9-]', '-', f"{prefix}-{meeting_id}")[:40]
    return f"{base}-{int(time.time())}"[:63]

# ==== A2I additions END ====


def _repair_case_json_with_llm(meeting_id: str, bad_text: str) -> str:
    """
    Ask the model to repair malformed JSON into a valid object matching CaseCheckPayload.
    Returns a JSON string (not dict). Raises on failure.
    """
    repair_prompt = JSON_REPAIR_PROMPT_TEMPLATE.format(bad_json=bad_text)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1200,
        "temperature": 0.0,
        "system": JSON_REPAIR_SYSTEM_MESSAGE,
        "messages": [{"role": "user", "content": [{"type": "text", "text": repair_prompt}]}],
    })
    raw_resp, latency_ms = helper.bedrock_infer(MODEL_ID, body)
    payload = json.loads(raw_resp)
    # Optional structured log
    helper.log_json("INFO", "LLM_REPAIR_OK", meetingId=meeting_id, latency_ms=latency_ms, input_chars=len(body), output_chars=len(raw_resp))

    text = "".join([b.get("text","") for b in payload.get("content", []) if b.get("type") == "text"]).strip()
    # try to extract the JSON block just in case
    try:
        return _extract_json_object(text)
    except Exception:
        return text  # already likely a pure JSON string
    
def _run_id() -> str:
    """
    Deterministic run id so re-runs under different versions separate cleanly.
    You can tweak to include git sha or build id if you have one.
    """
    # Example: v=1.1/model=sonnet-20240229/prompt=2025-09-03_144502Z
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"schema={SCHEMA_VERSION}/model={MODEL_VERSION}/prompt={PROMPT_VERSION}/ts={ts}"

def _summary_object_keys(meeting_id: str) -> tuple[str, str]:
    """
    Returns (run_key, latest_key) for summary JSON.
    - run_key: unique per run (includes versions + timestamp) - Athena partitioned
    - latest_key: stable pointer to the newest summary for this meeting
    """
    partitioner = get_s3_partitioner()
    timestamp = datetime.now(timezone.utc)

    # Check if Athena partitioning is enabled
    athena_partitioned = ATHENA_PARTITIONED

    if athena_partitioned:
        # Athena-optimized partitioned structure - save to data/ subdirectory
        # Format: summaries/data/version=1.2/year=2025/month=09/meeting_id=123/summary.json
        run_key = f"{S3_PREFIX}/data/version={SCHEMA_VERSION}/year={timestamp.year}/month={timestamp.month:02d}/meeting_id={meeting_id}/summary_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
        # Latest pointer in same partition
        latest_key = f"{S3_PREFIX}/data/version={SCHEMA_VERSION}/year={timestamp.year}/month={timestamp.month:02d}/meeting_id={meeting_id}/summary.json"
    else:
        # Legacy structure for backward compatibility
        run_id = _run_id()
        run_key = f"{S3_PREFIX}/meetings/{meeting_id}/{run_id}/summary.json"
        latest_key = f"{S3_PREFIX}/meetings/{meeting_id}/latest/summary.json"

    return run_key, latest_key


def run_case_check(meeting_id: str, cleaned_transcript: str, year: int = None, month: int = None) -> tuple[dict, str]:
    """Returns (payload_dict, s3_key).

    Args:
        meeting_id: The meeting identifier
        cleaned_transcript: The redacted transcript text
        year: Year for S3 partition (should match summary partition)
        month: Month for S3 partition (should match summary partition)
    """
    prompt = _build_case_prompt(meeting_id, cleaned_transcript, STARTER_SESSION_CHECKS)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 3000,  # increased to prevent JSON truncation
        "temperature": 0.2,
        # Extra guardrail to avoid prefaces:
        "system": CASE_CHECK_SYSTEM_MESSAGE,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    })

    raw_resp, latency_ms = helper.bedrock_infer(MODEL_ID, body)
    payload = json.loads(raw_resp)
    text = "".join([b.get("text","") for b in payload.get("content", []) if b.get("type") == "text"])
    helper.log_json("INFO", "LLM_CASECHECK_OK", meetingId=meeting_id, latency_ms=latency_ms, input_chars=len(body), output_chars=len(raw_resp))

    # Log raw LLM output for debugging
    helper.log_json("INFO", "RAW_CASE_LLM_OUTPUT", meetingId=meeting_id,
                   text_length=len(text), text_preview=text[:200], text_suffix=text[-200:])

    # 3-stage parse: raw -> strip fences -> extract {...} -> repair if needed
    try:
        parsed = CaseCheckPayload.model_validate_json(text)
    except ValidationError:
        try:
            cleaned = helper.strip_code_fences(text)
            parsed = CaseCheckPayload.model_validate_json(cleaned)
        except ValidationError:
            try:
                extracted = _extract_json_object(text)
                parsed = CaseCheckPayload.model_validate_json(extracted)
            except ValidationError:
                # One more chance: ask LLM to repair the JSON and parse again
                repaired = _repair_case_json_with_llm(meeting_id, text)
                parsed = CaseCheckPayload.model_validate_json(repaired)

    data = parsed.model_dump()

    # Hard-set authoritative fields (belt-and-braces)
    data["meeting_id"] = meeting_id
    data["model_version"] = MODEL_VERSION
    data["prompt_version"] = PROMPT_VERSION

    # Coerce None -> "" for stringy fields
    for r in data.get("results", []):
        r["evidence_quote"] = r.get("evidence_quote") or ""
        r["comment"] = r.get("comment") or ""
        # Clamp confidence to [0,1] scale
        try:
            r["confidence"] = max(0.0, min(1.0, float(r.get("confidence", 0.0))))
        except Exception:
            r["confidence"] = 0.0

    key = _save_case_json(meeting_id, data, year=year, month=month)
    return (data, key)

# ---------- Handler ----------
def lambda_handler(event, context):
    for record in event["Records"]:
        raw = record.get("body") or ""
        try:
            message = json.loads(raw)
        except Exception:
            # Skip malformed/empty messages instead of crashing the batch
            helper.log_json("WARN", "SKIP_INVALID_MESSAGE", body_len=len(raw))
            continue

        meeting_id = message.get("meetingId")
        coach_name = message.get("coachName")
        raw_emp = message.get("employerName")
        employer_name = raw_emp.strip() if isinstance(raw_emp, str) else None
        transcript = message.get("transcript")
        zoom_meeting_id = str(message.get("zoomMeetingId") or "").replace(" ", "")
        enable_case_check = bool(message.get("enableCaseCheck", False))
        force_reprocess = bool(message.get("forceReprocess", False))

        try:
            helper.log_json("INFO", "PIPELINE_START", meetingId=meeting_id, enableCaseCheck=enable_case_check, forceReprocess=force_reprocess)

            # Idempotency (handles SQS redeliveries)
            # Skip check if force_reprocess is enabled
            if not force_reprocess and job_is_completed(meeting_id):
                helper.log_json("INFO", "JOB_SKIPPED_ALREADY_COMPLETED", meetingId=meeting_id)
                continue

            # If force reprocess is enabled, reset status to allow overwriting
            if force_reprocess:
                helper.log_json("INFO", "FORCE_REPROCESS_ENABLED", meetingId=meeting_id)
                set_status(meeting_id, "PROCESSING", force=True)

                # Delete old summary and case check files to ensure fresh reprocessing
                helper.log_json("INFO", "DELETING_OLD_FILES", meetingId=meeting_id)

                # Try to get existing file paths from DynamoDB
                try:
                    resp = JOB_TABLE_REF.get_item(Key={"meetingId": meeting_id})
                    ddb_item = resp.get("Item", {})
                    old_summary_key = ddb_item.get("latestSummaryKey")
                    old_case_key = ddb_item.get("caseCheckKey")

                    # Delete old summary file
                    if old_summary_key:
                        try:
                            s3.delete_object(Bucket=SUMMARY_BUCKET, Key=old_summary_key)
                            helper.log_json("INFO", "DELETED_OLD_SUMMARY", meetingId=meeting_id, key=old_summary_key)
                        except Exception as e:
                            helper.log_json("WARN", "FAILED_DELETE_SUMMARY", meetingId=meeting_id, error=str(e))

                    # Delete old case check file
                    if old_case_key:
                        try:
                            s3.delete_object(Bucket=SUMMARY_BUCKET, Key=old_case_key)
                            helper.log_json("INFO", "DELETED_OLD_CASE_CHECK", meetingId=meeting_id, key=old_case_key)
                        except Exception as e:
                            helper.log_json("WARN", "FAILED_DELETE_CASE_CHECK", meetingId=meeting_id, error=str(e))

                    # Clear the keys from DynamoDB
                    JOB_TABLE_REF.update_item(
                        Key={"meetingId": meeting_id},
                        UpdateExpression="REMOVE latestSummaryKey, caseCheckKey, casePassRate, caseFailedCount",
                        ReturnValues="NONE"
                    )
                    helper.log_json("INFO", "CLEARED_OLD_METADATA", meetingId=meeting_id)

                except Exception as e:
                    helper.log_json("WARN", "FAILED_CLEANUP_OLD_FILES", meetingId=meeting_id, error=str(e))

            # Acquire transcript
            if transcript:
                source = "direct"
                if os.getenv("SAVE_TRANSCRIPTS", "false").lower() == "true":
                    now = datetime.now(timezone.utc)

                    if ATHENA_PARTITIONED:
                        # Save transcript to supplementary/ subdirectory
                        base = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={now.year}/month={now.month:02d}/meeting_id={meeting_id}"
                    else:
                        # Legacy format for backward compatibility
                        base = f"{S3_PREFIX}/{now:%Y}/{now:%m}/{meeting_id}"

                    _s3_put_text(SUMMARY_BUCKET, f"{base}/transcript.txt", transcript, "text/plain")

            elif zoom_meeting_id:
                set_status(
                    meeting_id, "FETCH_TRANSCRIPT", {"zoomMeetingId": zoom_meeting_id}
                )
                # STORE the fetched transcript to s3
                transcript, _ = fetch_transcript_by_zoom_meeting_id(zoom_meeting_id)
                source = "zoom_api"
            elif force_reprocess:
                # Force reprocess: fetch existing transcript from S3
                helper.log_json("INFO", "FETCHING_EXISTING_TRANSCRIPT", meetingId=meeting_id)
                set_status(meeting_id, "FETCH_TRANSCRIPT", {"source": "s3"})

                # Try multiple locations for transcript
                possible_paths = [
                    f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year=2025/month=09/meeting_id={meeting_id}/transcript.txt",
                    f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year=2025/month=10/meeting_id={meeting_id}/transcript.txt",
                    f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year=2025/month=08/meeting_id={meeting_id}/transcript.txt",
                    f"{S3_PREFIX}/version={SCHEMA_VERSION}/year=2025/month=09/meeting_id={meeting_id}/transcript.txt",
                ]

                transcript = None
                for path in possible_paths:
                    try:
                        response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=path)
                        transcript = response['Body'].read().decode('utf-8')
                        helper.log_json("INFO", "TRANSCRIPT_FOUND", meetingId=meeting_id, path=path)
                        break
                    except:
                        continue

                if not transcript:
                    raise ValueError(f"Force reprocess: Could not find existing transcript for meeting {meeting_id}")

                source = "s3_existing"
            else:
                raise ValueError("Provide either transcript text OR zoomMeetingId")

            # 0) Role normalisation
            transcript = normalise_roles(transcript, coach_name)

            # 1) PII redaction
            set_status(meeting_id, "PII_REDACTION")
            pii = pii_entities_chunked(transcript)
            redacted_transcript = redact_text(transcript, pii)

            # 2) LLM summary call (Claude 3 via Bedrock)
            helper.log_json("INFO", "STARTING_LLM_SUMMARY", meetingId=meeting_id, transcript_length=len(redacted_transcript))
            set_status(meeting_id, "LLM_SUMMARY")

            helper.log_json("INFO", "BUILDING_PROMPT", meetingId=meeting_id)
            try:
                prompt = build_prompt(redacted_transcript)
                helper.log_json("INFO", "PROMPT_BUILT", meetingId=meeting_id, prompt_length=len(prompt))
            except Exception as e:
                helper.log_json("ERROR", "PROMPT_BUILD_FAILED", meetingId=meeting_id,
                               error=str(e), error_type=type(e).__name__,
                               transcript_length=len(redacted_transcript))
                raise

            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1200,
                    "temperature": 0.3,
                    "system": SUMMARY_SYSTEM_MESSAGE,
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                }
            )

            helper.log_json("INFO", "CALLING_BEDROCK", meetingId=meeting_id,
                           body_length=len(body), prompt_length=len(prompt))

            raw_resp, latency_ms = helper.bedrock_infer(MODEL_ID, body)

            helper.log_json("INFO", "BEDROCK_CALL_SUCCESS", meetingId=meeting_id, latency_ms=latency_ms)

            # Log the raw Bedrock response for debugging
            helper.log_json("INFO", "BEDROCK_RAW_RESPONSE", meetingId=meeting_id,
                           raw_resp_length=len(raw_resp), raw_resp_preview=raw_resp[:300])

            try:
                payload = json.loads(raw_resp)
            except json.JSONDecodeError as e:
                helper.log_json("ERROR", "BEDROCK_JSON_PARSE_FAILED", meetingId=meeting_id,
                               error=str(e), raw_response=raw_resp[:500])
                raise ValueError(f"Invalid JSON response from Bedrock: {raw_resp[:200]}") from e

            text_blocks = [b.get("text","") for b in payload.get("content", []) if b.get("type") == "text"]
            raw_text = "".join(text_blocks)

            helper.log_json(
                "INFO",
                "LLM_SUMMARY_OK",
                meetingId=meeting_id,
                latency_ms=latency_ms,
                input_chars=len(body),
                output_chars=len(raw_resp),
            )

            # Log raw summary LLM output for debugging
            helper.log_json("INFO", "RAW_SUMMARY_LLM_OUTPUT", meetingId=meeting_id,
                           text_length=len(raw_text), text_preview=raw_text[:200], text_suffix=raw_text[-200:])

            set_status(meeting_id, "VALIDATING")
            try:
                # Extract the first top-level JSON object if the model added any preface
                helper.log_json("INFO", "EXTRACTING_JSON", meetingId=meeting_id, raw_text_length=len(raw_text))
                cleaned_json = _extract_json_object(raw_text)
                helper.log_json("INFO", "SUMMARY_EXTRACTED_JSON", meetingId=meeting_id,
                               extracted_length=len(cleaned_json), extracted_preview=cleaned_json[:200])
                helper.log_json("INFO", "VALIDATING_JSON", meetingId=meeting_id)
                summary_data = ClaudeResponse.model_validate_json(cleaned_json)
                helper.log_json("INFO", "JSON_VALIDATION_SUCCESS", meetingId=meeting_id)
            except ValidationError:
                # Save raw for debugging, then try again after stripping code fences
                debug_now = datetime.now(timezone.utc)

                if ATHENA_PARTITIONED:
                    # Use new Athena-partitioned format
                    raw_key = f"{S3_PREFIX}/version={SCHEMA_VERSION}/year={debug_now.year}/month={debug_now.month:02d}/meeting_id={meeting_id}/model_raw.txt"
                else:
                    # Legacy format for backward compatibility
                    raw_key = f"{S3_PREFIX}/{debug_now:%Y/%m}/{meeting_id}/model_raw.txt"

                s3.put_object(
                    Bucket=SUMMARY_BUCKET,
                    Key=raw_key,
                    Body=raw_text.encode("utf-8"),
                    ContentType="text/plain",
                )
                summary_data = ClaudeResponse.model_validate_json(
                    helper.strip_code_fences(raw_text)
                )


            # 4) Build summary payload, persist JSON
            payload_obj = _build_summary_payload(
                meeting_id=meeting_id, coach_name=coach_name, employer_name=employer_name, source=source, data=summary_data
            )
            helper.log_json("INFO", "PAYLOAD_DEBUG", meetingId=meeting_id, payload_employer=payload_obj["meeting"].get("employerName"))
            run_key, latest_key = _save_summary_json(meeting_id, payload_obj)


            # 5) Case checking (optional via request parameter)
            case_data = None
            case_key = None
            human_loop = None
            if enable_case_check:
                helper.log_json("INFO", "CASE_CHECK_ENABLED", meetingId=meeting_id)
                set_status(meeting_id, "CASE_CHECKING")

                # Extract year/month from summary key to ensure case check uses same partition
                year, month = None, None
                if ATHENA_PARTITIONED and latest_key:
                    import re
                    match = re.search(r'year=(\d+)/month=(\d+)', latest_key)
                    if match:
                        year = int(match.group(1))
                        month = int(match.group(2))
                        helper.log_json("INFO", "CASE_CHECK_PARTITION", meetingId=meeting_id, year=year, month=month)

                case_data, case_key = run_case_check(meeting_id, redacted_transcript, year=year, month=month)
                human_loop = start_case_review_if_needed(meeting_id, case_data, redacted_transcript)
            else:
                helper.log_json("INFO", "CASE_CHECK_DISABLED", meetingId=meeting_id)

            # 6) Mark completed with useful dashboard fields
            extra = {
                "summaryKey": run_key,
                "latestSummaryKey": latest_key,
                "sentiment": payload_obj["sentiment"]["label"],
                "actionCount": payload_obj["insights"]["action_count"],
                "source": source,
                "promptVersion": payload_obj["call_metadata"]["prompt_version"],
                "modelVersion": payload_obj["call_metadata"]["model_version"],
                "schemaVersion": payload_obj["call_metadata"]["schema_version"],
                "prefixVersion": S3_PREFIX,
                "employerName": payload_obj["meeting"].get("employerName")
            }
            if case_key:
                extra.update({
                    "caseCheckKey": case_key,
                    "casePassRate": case_data.get("overall", {}).get("pass_rate"),
                    "caseFailedCount": len(case_data.get("overall", {}).get("failed_ids", [])),
                })

            if human_loop:
                extra["a2iCaseLoop"] = human_loop
                set_status(meeting_id, "IN_REVIEW", extra)
            else:
                set_status(meeting_id, "COMPLETED", extra)

        except Exception as e:
            set_status(meeting_id, "FAILED", {"error": str(e)[:500]})
            helper.log_json("ERROR", "PIPELINE_FAILED", meetingId=meeting_id, error=str(e)[:500])

