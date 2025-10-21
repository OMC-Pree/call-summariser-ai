"""
Case Check Lambda - Step Functions workflow step
Performs compliance case checking using Bedrock Claude
"""
import json
import boto3
from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, ValidationError, Field
from datetime import datetime, timezone
from utils import helper
from utils.error_handler import lambda_error_handler
from constants import *
from prompts import CASE_CHECK_PROMPT_TEMPLATE, CASE_CHECK_SYSTEM_MESSAGE, JSON_REPAIR_PROMPT_TEMPLATE, JSON_REPAIR_SYSTEM_MESSAGE

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
s3 = boto3.client("s3")


def get_transcript_from_s3(s3_key: str) -> str:
    """Fetch transcript from S3"""
    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=s3_key)
    return response['Body'].read().decode('utf-8')


def chunk_transcript(transcript: str, chunk_size: int = 20000, overlap: int = 2000) -> List[str]:
    """
    Split transcript into overlapping chunks to ensure no compliance items are missed.
    Uses sentence boundaries to avoid splitting mid-statement.

    Args:
        transcript: Full transcript text
        chunk_size: Target size of each chunk in characters (~5000 tokens)
        overlap: Minimum number of characters to overlap between chunks

    Returns:
        List of transcript chunks
    """
    if len(transcript) <= chunk_size:
        return [transcript]

    import re
    chunks = []
    start = 0

    while start < len(transcript):
        end = min(start + chunk_size, len(transcript))

        # If not at the end, try to break at a sentence boundary
        if end < len(transcript):
            # Look for sentence endings within the last 500 chars of the chunk
            search_start = max(end - 500, start)
            remaining = transcript[search_start:end + 500]

            # Find the last sentence ending (.!?) followed by whitespace or newline
            sentence_endings = list(re.finditer(r'[.!?]\s+', remaining))
            if sentence_endings:
                # Use the last sentence ending found
                last_ending = sentence_endings[-1]
                end = search_start + last_ending.end()

        chunk = transcript[start:end]
        chunks.append(chunk)

        # Move start forward, accounting for overlap
        start = end - overlap

        # Break if we've covered the whole transcript
        if end >= len(transcript):
            break

    return chunks


def merge_case_check_results(chunk_results: List[dict]) -> dict:
    """
    Merge case check results from multiple chunks using conservative compliance strategy.

    Strategy (prioritizes compliance safety):
    - Fail > Pass > Inconclusive > NotApplicable
    - If ANY chunk shows Fail, the merged result is Fail (catches compliance issues)
    - If any chunk shows Pass (and no Fails), use Pass (evidence found)
    - Within same status, use highest confidence score
    - Combine evidence quotes from all relevant chunks

    This ensures we don't miss compliance violations even if they only appear in one chunk.
    """
    if not chunk_results:
        raise ValueError("No chunk results to merge")

    if len(chunk_results) == 1:
        return chunk_results[0]

    # Initialize with first result
    merged = {"results": []}

    # Get all check IDs from first result
    check_ids = [r["id"] for r in chunk_results[0]["results"]]

    # Merge each check across all chunks
    for check_id in check_ids:
        # Collect this check from all chunks
        check_across_chunks = []
        for chunk_result in chunk_results:
            for check in chunk_result["results"]:
                if check["id"] == check_id:
                    check_across_chunks.append(check)
                    break

        # Merge strategy for compliance safety:
        # - Fail takes priority (conservative approach - if ANY chunk shows failure, flag it)
        # - Then Pass (evidence found)
        # - Then Inconclusive
        # - Finally NotApplicable
        # Within same status, use highest confidence
        priority = {"Fail": 4, "Pass": 3, "Inconclusive": 2, "NotApplicable": 1}
        best_check = max(check_across_chunks, key=lambda c: (priority.get(c["status"], 0), c.get("confidence", 0)))

        # Combine evidence quotes from all chunks that found evidence
        all_quotes = [c.get("evidence_quote", "") for c in check_across_chunks
                     if c.get("evidence_quote") and c["status"] in ["Pass", "Fail"]]
        if all_quotes:
            best_check["evidence_quote"] = " | ".join(filter(None, all_quotes))[:500]  # Limit length

        merged["results"].append(best_check)

    # Copy metadata from first chunk
    merged["check_schema_version"] = chunk_results[0]["check_schema_version"]
    merged["session_type"] = chunk_results[0]["session_type"]
    merged["checklist_version"] = chunk_results[0]["checklist_version"]
    merged["meeting_id"] = chunk_results[0]["meeting_id"]
    merged["model_version"] = chunk_results[0]["model_version"]
    merged["prompt_version"] = chunk_results[0]["prompt_version"]

    # Recalculate overall statistics
    total_checks = len(merged["results"])
    passed_checks = sum(1 for r in merged["results"] if r["status"] == "Pass")
    pass_rate = passed_checks / total_checks if total_checks > 0 else 0

    failed_ids = [r["id"] for r in merged["results"] if r["status"] == "Fail"]
    high_severity_flags = [r["id"] for r in merged["results"]
                          if r["status"] == "Fail" and r.get("severity") == "high"]

    merged["overall"] = {
        "pass_rate": pass_rate,
        "failed_ids": failed_ids,
        "high_severity_flags": high_severity_flags,
        "has_high_severity_failures": len(high_severity_flags) > 0
    }

    return merged


# ---------- Case check models ----------
Span = Tuple[int, int]
Status = Literal["Pass", "Fail", "NotApplicable", "Inconclusive"]


class CaseCheckResult(BaseModel):
    id: str
    status: Status
    confidence: float
    evidence_spans: List[Span] = Field(default_factory=list)
    evidence_quote: Optional[str] = ""
    comment: Optional[str] = ""


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
    {"id": "call_recording_confirmed", "prompt": "Did the coach confirm call recording?", "required": True, "severity": "high"},
    {"id": "regulated_advice_given", "prompt": "Did the coach give regulated financial advice (e.g. specific product or investment recommendation)? NOTE: This is not permitted. If the coach gave regulated advice, the status must be 'Fail'.", "required": True, "severity": "high"},
    {"id": "dob_confirmed", "prompt": "Was the client's date of birth confirmed?", "required": True, "severity": "medium"},
    {"id": "full_name_confirmed", "prompt": "Was the client's full name confirmed?", "required": True, "severity": "medium"},
    {"id": "financial_info_confirmed", "prompt": "Was the client's financial information confirmed (income, debts, savings, etc.)?", "required": True, "severity": "medium"},
    {"id": "partner_status_confirmed", "prompt": "Was the client's partner/marital status confirmed?", "required": False, "severity": "low"},
    {"id": "dependents_confirmed", "prompt": "Were dependents confirmed?", "required": False, "severity": "low"},
    {"id": "vulnerability_identified", "prompt": "Was any client vulnerability identified or discussed?", "required": True, "severity": "medium"},
    {"id": "citizenship_residency_confirmed", "prompt": "Were the client's citizenship and residency confirmed?", "required": True, "severity": "medium"},
    {"id": "fees_charges_explained", "prompt": "Were fees and charges correctly explained to the client?", "required": True, "severity": "medium"},
    {"id": "client_agreed_to_sign_up", "prompt": "Did the client agree to sign up to the service?", "required": False, "severity": "medium"},
    {"id": "holistic_explanation", "prompt": "Did the coach explain how they and Octopus Money help clients holistically (including getting advice)?", "required": True, "severity": "medium"},
    {"id": "action_ownership_explained", "prompt": "Did the coach explain that success depends on actions the client must complete?", "required": True, "severity": "medium"},
    {"id": "goal_understood_and_addressed", "prompt": "Did the coach fully understand the client's most important goal and give relevant suggestions?", "required": True, "severity": "medium"},
    {"id": "services_understood", "prompt": "Did the client clearly understand all relevant areas where Octopus Money can help (upfront and ongoing)?", "required": True, "severity": "medium"},
    {"id": "suitability_for_ongoing_advice", "prompt": "Is the client suitable for ongoing advice?", "required": True, "severity": "high"}
]


def _extract_json_object(text: str) -> str:
    """Extract first top-level JSON object from text"""
    if not text:
        raise ValueError("Empty model output")
    t = text.strip()

    start = t.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

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
                    return t[start:i+1]

    return t[start:] if len(t[start:]) >= 20 else t


def _repair_case_json_with_llm(meeting_id: str, bad_text: str) -> str:
    """Ask LLM to repair malformed JSON"""
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
    helper.log_json("INFO", "LLM_REPAIR_OK", meetingId=meeting_id, latency_ms=latency_ms)

    text = "".join([b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"]).strip()

    try:
        return _extract_json_object(text)
    except Exception:
        return text


def _build_case_prompt(meeting_id: str, cleaned_transcript: str, checks: list) -> str:
    """Build case check prompt"""
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
    """Save case check JSON to S3"""
    if year is None or month is None:
        now = datetime.now(timezone.utc)
        year = now.year
        month = now.month

    if ATHENA_PARTITIONED:
        key = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={year}/month={month:02d}/meeting_id={meeting_id}/case_check.v{CASE_CHECK_SCHEMA_VERSION}.json"
    else:
        key = f"{S3_PREFIX}/{year:04d}/{month:02d}/{meeting_id}/case_check.v{CASE_CHECK_SCHEMA_VERSION}.json"

    s3.put_object(
        Bucket=SUMMARY_BUCKET,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )
    helper.log_json("INFO", "CASE_CHECK_SAVED", meetingId=meeting_id, s3Key=key)
    return key


@lambda_error_handler()
def lambda_handler(event, context):
    """
    Perform case check on redacted transcript.

    Input:
        - redactedTranscriptKey: str (S3 key to redacted transcript)
        - meetingId: str

    Output:
        - caseData: dict
        - caseKey: str
        - passRate: float
    """
    transcript_key = event.get("redactedTranscriptKey")
    meeting_id = event.get("meetingId")

    if not transcript_key:
        raise ValueError("redactedTranscriptKey is required")

    if not meeting_id:
        raise ValueError("meetingId is required")

    # Fetch transcript from S3
    full_transcript = get_transcript_from_s3(transcript_key)

    # Use chunking strategy for long transcripts
    CHUNK_SIZE = 20000  # ~5000 tokens per chunk (larger = fewer API calls)
    CHUNK_OVERLAP = 2000  # Larger overlap for compliance safety at boundaries

    chunks = chunk_transcript(full_transcript, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    num_chunks = len(chunks)

    if num_chunks > 1:
        helper.log_json("INFO", "USING_CHUNKED_CASE_CHECK",
                       meetingId=meeting_id,
                       transcript_length=len(full_transcript),
                       num_chunks=num_chunks,
                       chunk_size=CHUNK_SIZE)
    else:
        helper.log_json("INFO", "CASE_CHECK_START",
                       meetingId=meeting_id,
                       transcript_length=len(full_transcript))

    # Process each chunk
    chunk_results = []
    for idx, chunk in enumerate(chunks):
        helper.log_json("INFO", f"PROCESSING_CHUNK_{idx + 1}_OF_{num_chunks}",
                       meetingId=meeting_id,
                       chunk_length=len(chunk))

        # Build prompt for this chunk
        prompt = _build_case_prompt(meeting_id, chunk, STARTER_SESSION_CHECKS)

        # Call Bedrock for this chunk
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 6000,
            "temperature": 0.2,
            "system": CASE_CHECK_SYSTEM_MESSAGE,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        })

        raw_resp, latency_ms = helper.bedrock_infer(MODEL_ID, body)
        payload = json.loads(raw_resp)
        text = "".join([b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"])

        # Check if response was truncated
        stop_reason = payload.get("stop_reason")
        if stop_reason == "max_tokens":
            helper.log_json("ERROR", "CHUNK_CASECHECK_TRUNCATED",
                           meetingId=meeting_id,
                           chunk_idx=idx + 1,
                           message="Chunk response hit max_tokens - chunk may still be too large")
            raise ValueError(f"Case check chunk {idx + 1} truncated for meeting {meeting_id}")

        helper.log_json("INFO", f"CHUNK_{idx + 1}_LLM_OK",
                       meetingId=meeting_id,
                       latency_ms=latency_ms,
                       stop_reason=stop_reason)

        # Parse and validate chunk result
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
                    repaired = _repair_case_json_with_llm(meeting_id, text)
                    parsed = CaseCheckPayload.model_validate_json(repaired)

        chunk_results.append(parsed.model_dump())

    # Merge results from all chunks
    if num_chunks > 1:
        helper.log_json("INFO", "MERGING_CHUNK_RESULTS",
                       meetingId=meeting_id,
                       num_chunks=num_chunks)
        data = merge_case_check_results(chunk_results)
    else:
        data = chunk_results[0]

    # Normalize data
    data["meeting_id"] = meeting_id
    data["model_version"] = MODEL_VERSION
    data["prompt_version"] = PROMPT_VERSION

    severity_by_id = {c["id"]: c.get("severity", "low") for c in STARTER_SESSION_CHECKS}
    has_high_severity_failures = False

    for r in data.get("results", []):
        r["evidence_quote"] = r.get("evidence_quote") or ""
        r["comment"] = r.get("comment") or ""
        r["confidence"] = max(0.0, min(1.0, float(r.get("confidence", 0.0))))

        if r.get("status") == "Fail" and severity_by_id.get(r.get("id")) == "high":
            has_high_severity_failures = True

    data["overall"]["has_high_severity_failures"] = has_high_severity_failures

    # Save to S3
    key = _save_case_json(meeting_id, data)

    # Calculate pass rate
    pass_rate = float(data.get("overall", {}).get("pass_rate", 0.0))

    return {
        "caseData": data,
        "caseKey": key,
        "passRate": pass_rate
    }
