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


def get_summary_tool():
    """
    Create a tool definition for structured summary output.
    This ensures the LLM returns valid JSON matching our expected schema.
    """
    return {
        "toolSpec": {
            "name": "submit_call_summary",
            "description": "Submit the call summary with all required fields",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Overall summary of the call"
                        },
                        "key_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of key discussion points"
                        },
                        "action_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"}
                                },
                                "required": ["description"]
                            },
                            "description": "Action items from the call"
                        },
                        "sentiment_analysis": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "enum": ["Positive", "Neutral", "Negative"]
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                }
                            },
                            "required": ["label", "confidence"]
                        },
                        "themes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "label": {"type": "string"},
                                    "group": {"type": "string"},
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0
                                    },
                                    "evidence_quote": {
                                        "type": ["string", "null"]
                                    }
                                },
                                "required": ["id", "label", "confidence"]
                            },
                            "description": "Identified themes from the call (0-7 themes)"
                        }
                    },
                    "required": ["summary", "key_points", "action_items", "sentiment_analysis", "themes"]
                }
            }
        }
    }


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

    # Call Bedrock using Converse API with structured output
    messages = [
        {"role": "user", "content": [{"text": prompt}]}
    ]

    # Get tool definition for guaranteed JSON schema
    summary_tool = get_summary_tool()

    # Build system prompt
    # Note: Prompt caching disabled - not available in eu-west-2 region yet
    system_message = SUMMARY_SYSTEM_MESSAGE

    helper.log_json("INFO", "CALLING_BEDROCK", meetingId=meeting_id, prompt_length=len(prompt))

    resp, latency_ms = helper.bedrock_converse(
        model_id=MODEL_ID,
        messages=messages,
        system=system_message,  # Pass as string (no caching)
        max_tokens=1200,
        temperature=0.3,
        tools=[summary_tool]  # Force structured output
    )

    helper.log_json("INFO", "BEDROCK_CALL_SUCCESS", meetingId=meeting_id, latency_ms=latency_ms)

    # Extract structured output from tool use
    output_message = resp.get("output", {}).get("message", {})
    content_blocks = output_message.get("content", [])

    # Find the tool use block
    tool_use_block = None
    for block in content_blocks:
        if "toolUse" in block:
            tool_use_block = block["toolUse"]
            break

    if not tool_use_block:
        raise ValueError("No tool use block found in response")

    # Get validated JSON from tool input
    validated_json = tool_use_block["input"]
    raw_text = json.dumps(validated_json)

    # Log usage metrics from Converse API with cache information
    usage = resp.get("usage", {})
    log_data = {
        "meetingId": meeting_id,
        "latency_ms": latency_ms,
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "total_tokens": usage.get("totalTokens", 0),
        "output_chars": len(raw_text),
    }

    # Add cache metrics if available
    if "cacheReadInputTokens" in usage:
        log_data["cache_read_tokens"] = usage.get("cacheReadInputTokens", 0)
        log_data["cache_creation_tokens"] = usage.get("cacheCreationInputTokens", 0)
        cache_savings = usage.get("cacheReadInputTokens", 0) * 0.9
        log_data["estimated_cache_savings_tokens"] = int(cache_savings)

    helper.log_json("INFO", "LLM_SUMMARY_OK", **log_data)

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

    # Token usage already extracted from resp above (usage variable)
    token_usage = {
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "total_tokens": usage.get("totalTokens", 0)
    }

    return {
        "summaryKey": summary_key,  # S3 key only
        "tokenUsage": token_usage  # Metadata for tracking
    }
