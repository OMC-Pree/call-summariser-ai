"""
Summarise Lambda - Step Functions workflow step
Generates AI summary using Bedrock Claude via Converse API with Prompt Management
"""
import json
import os
from typing import Optional
from datetime import datetime, timezone

from utils import helper
from utils.error_handler import lambda_error_handler, ValidationError
from utils.prompt_management import invoke_with_prompt_management, get_prompt_arn_from_parameter_store
from utils.aws_clients import AWSClients
from constants import *

# Use centralized AWS clients
bedrock = AWSClients.bedrock_runtime()
s3 = AWSClients.s3()

# Prompt Management configuration
USE_PROMPT_MANAGEMENT = os.getenv("USE_PROMPT_MANAGEMENT", "true").lower() == "true"
PROMPT_PARAM_NAME = os.getenv("PROMPT_PARAM_NAME_SUMMARY", "/call-summariser/prompts/summary/current")

# Cache for prompt ARN (loaded once per Lambda container)
_prompt_arn_cache = {}


def get_prompt_arn() -> Optional[str]:
    """Get prompt ARN from Parameter Store (cached for Lambda container lifetime)"""
    return get_prompt_arn_from_parameter_store(
        param_name=PROMPT_PARAM_NAME,
        cache_dict=_prompt_arn_cache,
        use_prompt_management=USE_PROMPT_MANAGEMENT
    )


def get_summary_tool():
    """
    Create a tool definition for structured summary output.
    SIMPLIFIED: Reduced complexity to improve reliability with Claude 3 Sonnet's 4K token limit.
    All arrays now use simple strings instead of nested objects.
    """
    return {
        "toolSpec": {
            "name": "submit_call_summary",
            "description": "Submit the call summary with all required fields. Keep responses concise.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Overall summary of the call (2-3 paragraphs maximum)"
                        },
                        "key_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of 5-7 key discussion points as separate strings (NOT formatted text, each point is one array element)"
                        },
                        "action_items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of action items from the call (simple strings, not objects)"
                        },
                        "sentiment_analysis": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "enum": ["Positive", "Neutral", "Negative"],
                                    "description": "Overall sentiment of the call"
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Confidence in sentiment assessment"
                                }
                            },
                            "required": ["label", "confidence"]
                        },
                        "themes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": "Theme identifier (lowercase, underscored)"
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": "Human-readable theme name"
                                    },
                                    "group": {
                                        "type": "string",
                                        "description": "Theme category (e.g., financial_goals, risk_management, debt_management)"
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                        "description": "Confidence score for theme identification"
                                    }
                                },
                                "required": ["id", "label", "group", "confidence"]
                            },
                            "description": "List of 3-5 main themes with categorization and confidence (0-5 themes max)"
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


def build_prompt_variables(transcript: str) -> dict:
    """Build variables for prompt template"""
    return {"transcript": transcript}


@lambda_error_handler()
def lambda_handler(event, context):
    """
    Generate AI summary from redacted transcript using Bedrock Claude.

    Input:
        - redactedTranscriptKey: str (S3 key to redacted transcript)
        - meetingId: str
        - forceReprocess: bool (optional, default False)

    Output:
        - summaryKey: str (S3 key to raw LLM output for validation step)
        - tokenUsage: dict (token usage metadata)
    """
    transcript_key = event.get("redactedTranscriptKey")
    meeting_id = event.get("meetingId")
    force_reprocess = event.get("forceReprocess", False)

    if not transcript_key:
        raise ValidationError("redactedTranscriptKey is required")

    if not meeting_id:
        raise ValidationError("meetingId is required")

    # Determine expected summary S3 key
    if ATHENA_PARTITIONED:
        now = datetime.now(timezone.utc)
        summary_key = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={now.year}/month={now.month:02d}/meeting_id={meeting_id}/raw_summary.json"
    else:
        summary_key = f"{S3_PREFIX}/supplementary/{meeting_id}/raw_summary.json"

    # Idempotency check: If summary already exists and not forcing reprocess, return it
    if not force_reprocess:
        try:
            s3.head_object(Bucket=SUMMARY_BUCKET, Key=summary_key)
            helper.log_json("INFO", "SUMMARY_EXISTS", meetingId=meeting_id, summaryKey=summary_key, reused=True)
            return {
                "summaryKey": summary_key,
                "tokenUsage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            }
        except s3.exceptions.NoSuchKey:
            # File doesn't exist, proceed with generation
            pass
        except Exception as e:
            # Log error but continue with processing
            helper.log_json("WARN", "SUMMARY_CHECK_FAILED", meetingId=meeting_id, error=str(e))

    # Fetch transcript from S3
    transcript = get_transcript_from_s3(transcript_key)

    helper.log_json("INFO", "STARTING_LLM_SUMMARY",
                   meetingId=meeting_id,
                   transcript_length=len(transcript),
                   use_prompt_management=USE_PROMPT_MANAGEMENT)

    # Get tool definition for guaranteed JSON schema
    summary_tool = get_summary_tool()

    # Build prompt variables
    variables = build_prompt_variables(transcript)

    # Get prompt ARN (lazy-loaded, cached)
    prompt_arn = get_prompt_arn()

    helper.log_json("INFO", "CALLING_BEDROCK",
                   meetingId=meeting_id,
                   prompt_arn=prompt_arn,
                   use_prompt_management=USE_PROMPT_MANAGEMENT)

    # Call Bedrock with Prompt Management or fallback to inline
    if prompt_arn:
        # Fetch prompt from Prompt Management and add tools at runtime
        resp, latency_ms = invoke_with_prompt_management(
            prompt_arn=prompt_arn,
            variables=variables,
            model_id=MODEL_ID,
            tools=[summary_tool],
            tool_choice={"tool": {"name": "submit_call_summary"}},
            system_override=None  # Use system prompt from Prompt Management
        )
    else:
        # Fallback to inline prompt (for backward compatibility)
        from prompts import SUMMARY_PROMPT_TEMPLATE, SUMMARY_SYSTEM_MESSAGE
        prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript)
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        resp, latency_ms = helper.bedrock_converse(
            model_id=MODEL_ID,
            messages=messages,
            system=SUMMARY_SYSTEM_MESSAGE,
            max_tokens=1200,
            temperature=0.3,
            tools=[summary_tool],
            tool_choice={"tool": {"name": "submit_call_summary"}}
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

    # Debug logging: log the raw tool output structure
    helper.log_json("DEBUG", "RAW_TOOL_OUTPUT",
                   meetingId=meeting_id,
                   has_summary="summary" in validated_json,
                   has_key_points="key_points" in validated_json,
                   has_sentiment_analysis="sentiment_analysis" in validated_json,
                   key_points_type=type(validated_json.get("key_points")).__name__ if "key_points" in validated_json else "missing",
                   key_points_value_preview=str(validated_json.get("key_points", ""))[:100])

    # Defensive parsing: Claude sometimes returns JSON array fields as strings when output is large
    # Parse any stringified array fields back to proper arrays
    validated_json = helper.parse_stringified_fields(
        data=validated_json,
        fields=["key_points", "action_items", "themes"],
        meeting_id=meeting_id,
        context="summary"
    )

    # Transform action_items from string array to object array for backward compatibility
    # ["Action 1", "Action 2"] -> [{"description": "Action 1"}, {"description": "Action 2"}]
    if "action_items" in validated_json and validated_json["action_items"]:
        if isinstance(validated_json["action_items"], list) and len(validated_json["action_items"]) > 0:
            if isinstance(validated_json["action_items"][0], str):
                validated_json["action_items"] = [
                    {"description": item} for item in validated_json["action_items"]
                ]

    # Serialize for S3 storage
    raw_text = json.dumps(validated_json, ensure_ascii=False, indent=2)

    # Calculate cost and log usage metrics
    usage = resp.get("usage", {})
    cost_breakdown = helper.calculate_bedrock_cost(usage, model_id="claude-3-7-sonnet")

    log_data = {
        "meetingId": meeting_id,
        "operation": "summary",
        "latency_ms": latency_ms,
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "total_tokens": usage.get("totalTokens", 0),
        "output_chars": len(raw_text),
        "cost_usd": cost_breakdown["total_cost"],
        "input_cost_usd": cost_breakdown["input_cost"],
        "output_cost_usd": cost_breakdown["output_cost"]
    }

    # Add cache metrics if available
    if "cacheReadInputTokens" in usage:
        log_data["cache_read_tokens"] = usage.get("cacheReadInputTokens", 0)
        log_data["cache_creation_tokens"] = usage.get("cacheCreationInputTokens", 0)
        log_data["cache_read_cost_usd"] = cost_breakdown["cache_read_cost"]
        log_data["cache_write_cost_usd"] = cost_breakdown["cache_write_cost"]
        log_data["cache_savings_usd"] = cost_breakdown["cache_savings"]

    helper.log_json("INFO", "LLM_SUMMARY_OK", **log_data)

    # Save raw summary to supplementary folder for validation step (summary_key already determined earlier)
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
