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
# JSON repair prompts no longer needed with structured output

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


# _extract_json_object and _repair_json_with_llm removed - no longer needed with structured output


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


@lambda_error_handler()
def lambda_handler(event, context):
    """
    Validate LLM JSON output (simplified for structured output via Tool Use).

    Since we now use Tool Use for structured output, the JSON is guaranteed to be valid.
    This function now simply validates with Pydantic and saves to S3.

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

    # Load structured output from S3
    helper.log_json("INFO", "LOADING_SUMMARY_FROM_S3", meetingId=meeting_id, summaryKey=summary_key)
    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=summary_key)
    raw_response = response['Body'].read().decode('utf-8')

    helper.log_json("INFO", "VALIDATING_STRUCTURED_OUTPUT", meetingId=meeting_id, validationType=validation_type)

    # With Tool Use, JSON is already validated by the API
    # Just parse and validate with Pydantic for type safety
    try:
        validated = ClaudeResponse.model_validate_json(raw_response)
        helper.log_json("INFO", "VALIDATION_SUCCESS", meetingId=meeting_id)

        validated_key = _save_validated_to_s3(meeting_id, validated.model_dump())
        return {
            "validatedDataKey": validated_key,
            "isValid": True
        }
    except ValidationError as e:
        # This should never happen with Tool Use, but handle gracefully
        # Log the actual response to debug
        helper.log_json("ERROR", "VALIDATION_FAILED_UNEXPECTED",
                       meetingId=meeting_id,
                       error=str(e)[:500],
                       raw_response_preview=raw_response[:500],
                       message="Structured output validation failed - this should not happen with Tool Use")

        # Always try to save and return a key to avoid breaking the workflow
        try:
            # Parse the JSON (might be valid JSON but wrong schema)
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            # If even JSON parsing fails, create minimal valid response
            helper.log_json("ERROR", "INVALID_JSON_IN_RESPONSE",
                           meetingId=meeting_id,
                           message="Response is not valid JSON")
            data = {
                "summary": "Error: Invalid response from LLM",
                "key_points": [],
                "action_items": [],
                "sentiment_analysis": {"label": "Neutral", "confidence": 0.0},
                "themes": [],
                "error": str(e)[:200]
            }

        # Save whatever we have
        validated_key = _save_validated_to_s3(meeting_id, data)
        helper.log_json("WARNING", "SAVED_DESPITE_VALIDATION_ERROR",
                       meetingId=meeting_id,
                       validatedKey=validated_key)

        # Always return a valid response to avoid breaking the workflow
        return {
            "validatedDataKey": validated_key,
            "isValid": False,
            "validationError": str(e)[:200]
        }
