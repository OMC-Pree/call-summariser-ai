"""
Summarise Lambda - Step Functions workflow step
Generates AI summary using Bedrock Claude via Converse API (streaming off)
"""
import json
import boto3
from utils import helper
from utils.error_handler import lambda_error_handler, ValidationError
from constants import *
from prompts import SUMMARY_PROMPT_TEMPLATE, SUMMARY_SYSTEM_MESSAGE

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
s3 = boto3.client("s3")


def get_transcript_from_s3(s3_key: str) -> str:
    """Fetch transcript from S3"""
    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=s3_key)
    return response['Body'].read().decode('utf-8')


def build_prompt(transcript: str) -> str:
    """Build the summary prompt from transcript"""
    return SUMMARY_PROMPT_TEMPLATE.replace("{transcript}", transcript)


@lambda_error_handler()
def lambda_handler(event, context):
    """
    Generate AI summary from redacted transcript using Bedrock Claude.

    Input:
        - redactedTranscriptKey: str (S3 key to redacted transcript)
        - meetingId: str

    Output:
        - summary: dict (parsed response preview)
        - rawResponse: str (full LLM response for validation step)
    """
    transcript_key = event.get("redactedTranscriptKey")
    meeting_id = event.get("meetingId")

    if not transcript_key:
        raise ValidationError("redactedTranscriptKey is required")

    if not meeting_id:
        raise ValidationError("meetingId is required")

    # Fetch transcript from S3
    transcript = get_transcript_from_s3(transcript_key)

    helper.log_json("INFO", "STARTING_LLM_SUMMARY", meetingId=meeting_id, transcript_length=len(transcript))

    # Build prompt
    prompt = build_prompt(transcript)

    # Call Bedrock
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1200,
        "temperature": 0.3,
        "system": SUMMARY_SYSTEM_MESSAGE,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    })

    helper.log_json("INFO", "CALLING_BEDROCK", meetingId=meeting_id, body_length=len(body))

    raw_resp, latency_ms = helper.bedrock_infer(MODEL_ID, body)

    helper.log_json("INFO", "BEDROCK_CALL_SUCCESS", meetingId=meeting_id, latency_ms=latency_ms)

    # Parse response
    payload = json.loads(raw_resp)
    text_blocks = [b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"]
    raw_text = "".join(text_blocks)

    helper.log_json(
        "INFO",
        "LLM_SUMMARY_OK",
        meetingId=meeting_id,
        latency_ms=latency_ms,
        input_chars=len(body),
        output_chars=len(raw_resp),
    )

    return {
        "summary": {
            "preview": raw_text[:500],  # Preview for logging
            "length": len(raw_text)
        },
        "rawResponse": raw_text  # Full response for validation
    }
