"""
Validate & Repair Lambda - Step Functions workflow step
Validates LLM output with Pydantic; uses LLM JSON-repair as fallback
"""
import json
import boto3
from typing import List, Optional
from pydantic import BaseModel, ValidationError, Field
from utils import helper
from utils.error_handler import lambda_error_handler
from constants import *
from prompts import JSON_REPAIR_PROMPT_TEMPLATE, JSON_REPAIR_SYSTEM_MESSAGE

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
s3 = boto3.client("s3")


# ---------- Summary response models ----------
class ActionItem(BaseModel):
    description: str


class SentimentAnalysis(BaseModel):
    label: str
    confidence: float


class ThemeItem(BaseModel):
    id: str
    label: str
    group: Optional[str] = "General"
    confidence: float
    evidence_quote: Optional[str] = None


class ClaudeResponse(BaseModel):
    summary: str
    key_points: List[str]
    action_items: List[ActionItem]
    sentiment_analysis: SentimentAnalysis
    themes: List[ThemeItem] = Field(default_factory=list)


def _extract_json_object(text: str) -> str:
    """
    Extract the first top-level {...} JSON object from text.
    Uses proper brace counting to handle nested structures correctly.
    """
    if not text:
        raise ValueError("Empty model output")
    t = text.strip()

    start = t.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

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

    incomplete_text = t[start:]
    if len(incomplete_text) < 20 or not incomplete_text.strip().startswith('{'):
        return t
    return incomplete_text


def _save_validated_to_s3(meeting_id: str, validated_data: dict) -> str:
    """Save validated data to S3 and return the key"""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    if ATHENA_PARTITIONED:
        validated_key = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={now.year}/month={now.month:02d}/meeting_id={meeting_id}/validated_summary.json"
    else:
        validated_key = f"{S3_PREFIX}/supplementary/{meeting_id}/validated_summary.json"

    validated_json = json.dumps(validated_data)
    s3.put_object(
        Bucket=SUMMARY_BUCKET,
        Key=validated_key,
        Body=validated_json.encode('utf-8'),
        ContentType='application/json'
    )

    helper.log_json("INFO", "VALIDATED_DATA_SAVED", meetingId=meeting_id, validatedKey=validated_key, size=len(validated_json))
    return validated_key


def _repair_json_with_llm(meeting_id: str, bad_text: str) -> str:
    """
    Ask the model to repair malformed JSON.
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
    helper.log_json("INFO", "LLM_REPAIR_OK", meetingId=meeting_id, latency_ms=latency_ms)

    text = "".join([b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"]).strip()

    try:
        return _extract_json_object(text)
    except Exception:
        return text


@lambda_error_handler()
def lambda_handler(event, context):
    """
    Validate and repair LLM JSON output.

    Input:
        - summaryKey: str (S3 key to raw LLM output)
        - meetingId: str
        - validationType: str (summary|casecheck)

    Output:
        - validatedDataKey: str (S3 key to validated summary)
        - isValid: bool
    """
    summary_key = event.get("summaryKey")
    meeting_id = event.get("meetingId")
    validation_type = event.get("validationType", "summary")

    if not summary_key:
        raise ValueError("summaryKey is required")

    if not meeting_id:
        raise ValueError("meetingId is required")

    # Load raw response from S3
    helper.log_json("INFO", "LOADING_SUMMARY_FROM_S3", meetingId=meeting_id, summaryKey=summary_key)
    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=summary_key)
    raw_response = response['Body'].read().decode('utf-8')

    helper.log_json("INFO", "VALIDATING", meetingId=meeting_id, validationType=validation_type)

    # Try to validate directly
    try:
        cleaned_json = _extract_json_object(raw_response)
        validated = ClaudeResponse.model_validate_json(cleaned_json)
        helper.log_json("INFO", "VALIDATION_SUCCESS_DIRECT", meetingId=meeting_id)

        validated_key = _save_validated_to_s3(meeting_id, validated.model_dump())
        return {
            "validatedDataKey": validated_key,
            "isValid": True
        }
    except ValidationError as e:
        helper.log_json("WARN", "VALIDATION_FAILED_FIRST_ATTEMPT", meetingId=meeting_id, error=str(e)[:200])

    # Try stripping code fences
    try:
        cleaned = helper.strip_code_fences(raw_response)
        validated = ClaudeResponse.model_validate_json(cleaned)
        helper.log_json("INFO", "VALIDATION_SUCCESS_AFTER_STRIP", meetingId=meeting_id)

        validated_key = _save_validated_to_s3(meeting_id, validated.model_dump())
        return {
            "validatedDataKey": validated_key,
            "isValid": True
        }
    except ValidationError as e:
        helper.log_json("WARN", "VALIDATION_FAILED_AFTER_STRIP", meetingId=meeting_id, error=str(e)[:200])

    # Try extracting JSON object
    try:
        extracted = _extract_json_object(raw_response)
        validated = ClaudeResponse.model_validate_json(extracted)
        helper.log_json("INFO", "VALIDATION_SUCCESS_AFTER_EXTRACT", meetingId=meeting_id)

        validated_key = _save_validated_to_s3(meeting_id, validated.model_dump())
        return {
            "validatedDataKey": validated_key,
            "isValid": True
        }
    except ValidationError as e:
        helper.log_json("WARN", "VALIDATION_FAILED_AFTER_EXTRACT", meetingId=meeting_id, error=str(e)[:200])

    # Last resort: LLM repair
    try:
        from datetime import datetime, timezone

        # Save raw for debugging in supplementary folder
        if ATHENA_PARTITIONED:
            now = datetime.now(timezone.utc)
            raw_key = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={now.year}/month={now.month:02d}/meeting_id={meeting_id}/model_raw.txt"
        else:
            raw_key = f"{S3_PREFIX}/supplementary/{meeting_id}/model_raw.txt"

        s3.put_object(
            Bucket=SUMMARY_BUCKET,
            Key=raw_key,
            Body=raw_response.encode("utf-8"),
            ContentType="text/plain",
        )

        repaired = _repair_json_with_llm(meeting_id, raw_response)
        validated = ClaudeResponse.model_validate_json(repaired)
        helper.log_json("INFO", "VALIDATION_SUCCESS_AFTER_REPAIR", meetingId=meeting_id)

        validated_key = _save_validated_to_s3(meeting_id, validated.model_dump())
        return {
            "validatedDataKey": validated_key,
            "isValid": True
        }
    except Exception as e:
        helper.log_json("ERROR", "VALIDATION_FAILED_ALL_ATTEMPTS", meetingId=meeting_id, error=str(e)[:500])
        raise ValueError(f"Failed to validate JSON after all repair attempts: {str(e)[:200]}")
