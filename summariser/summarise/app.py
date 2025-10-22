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
from datetime import datetime, timezone

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
        - summaryKey: str (S3 key to raw LLM output for validation step)
        - tokenUsage: dict (token usage metadata)
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

    # Save raw summary to supplementary folder for validation step
    if ATHENA_PARTITIONED:
        now = datetime.now(timezone.utc)
        summary_key = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={now.year}/month={now.month:02d}/meeting_id={meeting_id}/raw_summary.json"
    else:
        summary_key = f"{S3_PREFIX}/supplementary/{meeting_id}/raw_summary.json"
    s3.put_object(
        Bucket=SUMMARY_BUCKET,
        Key=summary_key,
        Body=raw_text,
        ContentType='application/json'
    )

    helper.log_json("INFO", "RAW_SUMMARY_SAVED",
                   meetingId=meeting_id,
                   summaryKey=summary_key,
                   size=len(raw_text))

    # Extract token usage from payload
    usage = payload.get("usage", {})
    token_usage = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    }

    return {
        "summaryKey": summary_key,  # S3 key only
        "tokenUsage": token_usage  # Metadata for tracking
    }
