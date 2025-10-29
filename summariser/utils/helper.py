import json
import random
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from botocore.exceptions import ClientError
from botocore.config import Config
import boto3
from constants import AWS_REGION

# AWS client with timeout configuration
# read_timeout: Maximum time to wait for Bedrock to return a response (important for long inference)
# connect_timeout: Maximum time to wait for connection to Bedrock
bedrock_config = Config(
    read_timeout=180,  # 3 minutes max for Bedrock inference per chunk
    connect_timeout=10,
    retries={'max_attempts': 0}  # We handle retries manually in bedrock_converse()
)
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=bedrock_config)

# Claude 3.7 Sonnet pricing (AWS Bedrock, as of 2025)
BEDROCK_PRICING = {
    "claude-3-7-sonnet": {
        "input_per_1k": 0.003,      # $3.00 per 1M tokens
        "output_per_1k": 0.015,     # $15.00 per 1M tokens
        "cache_read_per_1k": 0.0003,    # $0.30 per 1M tokens (90% cheaper)
        "cache_write_per_1k": 0.00375   # $3.75 per 1M tokens
    }
}


def calculate_bedrock_cost(usage: dict, model_id: str = "claude-3-7-sonnet") -> dict:
    """
    Calculate AWS Bedrock API cost from token usage.

    Args:
        usage: Usage dict from Bedrock response with token counts
        model_id: Model identifier for pricing lookup

    Returns:
        Dict with cost breakdown and total
    """
    pricing = BEDROCK_PRICING.get(model_id, BEDROCK_PRICING["claude-3-7-sonnet"])

    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    cache_read_tokens = usage.get("cacheReadInputTokens", 0)
    cache_write_tokens = usage.get("cacheCreationInputTokens", 0)

    # Calculate costs
    input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
    output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
    cache_read_cost = (cache_read_tokens / 1000) * pricing["cache_read_per_1k"]
    cache_write_cost = (cache_write_tokens / 1000) * pricing["cache_write_per_1k"]

    total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

    # Calculate savings from caching (vs non-cached input)
    cache_savings = 0
    if cache_read_tokens > 0:
        non_cached_cost = (cache_read_tokens / 1000) * pricing["input_per_1k"]
        cache_savings = non_cached_cost - cache_read_cost

    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "cache_read_cost": round(cache_read_cost, 6),
        "cache_write_cost": round(cache_write_cost, 6),
        "cache_savings": round(cache_savings, 6),
        "total_cost": round(total_cost, 6)
    }


def log_json(level: str, msg: str, **kwargs):
    """
    Print a single JSON line for CloudWatch logs.
    Usage: log_json("INFO", "LLM_SUMMARY_OK", meetingId=..., latency_ms=..., input_chars=..., output_chars=...)
    """
    try:
        payload = {
            "level": level.upper(),
            "message": msg,
            "ts": datetime.now(timezone.utc).isoformat() + "Z",
        }
        if kwargs:
            payload.update(kwargs)
        print(json.dumps(payload, ensure_ascii=False))
    except Exception:
        # never fail logging
        print(f"{level.upper()} {msg} {kwargs}")

def _should_retry_bedrock_error(err: Exception) -> bool:
    if isinstance(err, ClientError):
        code = err.response.get("Error", {}).get("Code", "")
        return code in {
            "ThrottlingException",
            "ModelTimeoutException",
            "ServiceUnavailableException",
            "InternalServerException",
            "BandwidthLimitExceeded",
        }
    # Fallback: no retry
    return False

def bedrock_converse(
    model_id: str,
    messages: list,
    system: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    tools: list = None,
    tool_choice: dict = None,
    tries: int = 5,
    base: float = 0.6,
    max_sleep: float = 6.0
):
    """
    Invoke Bedrock using Converse API with exponential backoff + jitter.

    Args:
        model_id: Bedrock model ID
        messages: List of message dicts with 'role' and 'content'
        system: Optional system prompt - can be string or list of system blocks
                String: Simple text system prompt
                List: System blocks with optional cache control, e.g.:
                      [{"text": "...", "cacheControl": {"type": "ephemeral"}}]
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        tools: Optional list of tool definitions for structured output
        tool_choice: Optional tool choice configuration (e.g., {"auto": {}} or {"tool": {"name": "tool_name"}})
        tries: Number of retry attempts
        base: Base delay for exponential backoff
        max_sleep: Maximum sleep duration

    Returns:
        Tuple of (response_dict, latency_ms)
        response_dict contains: output, stopReason, usage, etc.
    """
    last_err = None
    for attempt in range(1, tries + 1):
        t0 = time.perf_counter()
        try:
            # Build request parameters
            request_params = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                }
            }

            # Add system prompt if provided
            if system:
                # Support both string and list formats
                if isinstance(system, str):
                    request_params["system"] = [{"text": system}]
                elif isinstance(system, list):
                    request_params["system"] = system
                else:
                    raise ValueError(f"system must be str or list, got {type(system)}")

            # Add tools if provided (for structured output)
            if tools:
                request_params["toolConfig"] = {"tools": tools}

                # Add tool choice if specified, otherwise force tool use
                if tool_choice:
                    request_params["toolConfig"]["toolChoice"] = tool_choice
                elif len(tools) == 1:
                    # If only one tool, force its use for guaranteed structured output
                    request_params["toolConfig"]["toolChoice"] = {
                        "tool": {"name": tools[0]["toolSpec"]["name"]}
                    }

            # Call Converse API
            resp = bedrock.converse(**request_params)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return resp, latency_ms

        except Exception as e:
            last_err = e
            retryable = _should_retry_bedrock_error(e)
            # Also retry on transient network issues
            if not retryable and not isinstance(e, TimeoutError):
                # If not clearly retryable, only retry first time as grace
                if attempt >= 2:
                    break
            # backoff with jitter
            sleep_s = min(max_sleep, base * (2 ** (attempt - 1))) * (0.7 + 0.6 * random.random())
            time.sleep(sleep_s)
    raise last_err


def parse_stringified_fields(
    data: dict,
    fields: list,
    meeting_id: str,
    context: str = ""
) -> dict:
    """
    Defensive parsing for LLM tool outputs that may stringify array/object fields.

    Claude's Converse API with Tool Use sometimes returns array fields as JSON strings
    instead of parsed arrays when output is large or complex. This function detects
    and parses those stringified fields back to their proper types.

    Args:
        data: The validated_json from tool use (typically tool_use_block["input"])
        fields: List of field names that should be arrays/objects but might be strings
        meeting_id: Meeting ID for logging context
        context: Additional context for logging (e.g., "chunk_1", "summary")

    Returns:
        Modified data dict with parsed fields

    Raises:
        ValueError: If a field is a string but cannot be parsed as valid JSON

    Example:
        >>> tool_output = tool_use_block["input"]
        >>> tool_output = parse_stringified_fields(
        ...     tool_output,
        ...     ["results", "key_points"],
        ...     meeting_id="123",
        ...     context="chunk_1"
        ... )
    """
    for field in fields:
        if field not in data:
            continue

        if isinstance(data[field], str):
            # Skip empty strings - treat as empty array/object
            stripped = data[field].strip()
            if not stripped:
                log_json("WARNING", "FIELD_WAS_EMPTY_STRING",
                        meetingId=meeting_id,
                        field=field,
                        context=context or "unknown",
                        message=f"{field} field was empty string, setting to empty array")
                data[field] = []
                continue

            # Check if it's a bullet-point formatted string (LLM sometimes does this)
            # If it starts with "- " or "• ", it's likely a formatted list, not JSON
            if stripped.startswith(("- ", "• ", "* ")):
                # Split by newlines and extract bullet points
                lines = [line.strip() for line in stripped.split('\n') if line.strip()]
                # Remove bullet markers and convert to array
                parsed_array = []
                for line in lines:
                    # Remove common bullet markers
                    cleaned = line.lstrip('•*-— ').strip()
                    if cleaned:
                        parsed_array.append(cleaned)

                data[field] = parsed_array
                log_json("WARNING", "FIELD_WAS_FORMATTED_STRING",
                        meetingId=meeting_id,
                        field=field,
                        context=context or "unknown",
                        items_parsed=len(parsed_array),
                        message=f"{field} was bullet-formatted string, converted to array")
                continue

            # Try to parse as JSON
            try:
                data[field] = json.loads(stripped)
                log_json("WARNING", "FIELD_WAS_STRING",
                        meetingId=meeting_id,
                        field=field,
                        context=context or "unknown",
                        message=f"{field} field was returned as string, successfully parsed")
            except json.JSONDecodeError as e:
                # Check if it's multiple line-separated JSON arrays
                # Claude sometimes returns: ["item1"]\n["item2"]\n["item3"]
                if '\n' in stripped and "Extra data" in str(e):
                    try:
                        merged_array = []
                        for line in stripped.split('\n'):
                            line = line.strip()
                            if line:
                                parsed_line = json.loads(line)
                                if isinstance(parsed_line, list):
                                    merged_array.extend(parsed_line)
                                else:
                                    merged_array.append(parsed_line)
                        data[field] = merged_array
                        log_json("WARNING", "FIELD_WAS_MULTILINE_JSON",
                                meetingId=meeting_id,
                                field=field,
                                context=context or "unknown",
                                items_merged=len(merged_array),
                                message=f"{field} was multiple line-separated JSON arrays, merged successfully")
                        continue
                    except Exception:
                        # If merge fails, fall through to original error
                        pass

                log_json("ERROR", "FIELD_PARSE_FAILED",
                        meetingId=meeting_id,
                        field=field,
                        context=context or "unknown",
                        error=str(e),
                        raw_value_preview=str(data[field])[:500])
                context_msg = f" in {context}" if context else ""
                raise ValueError(f"Failed to parse stringified {field} field{context_msg}: {str(e)}")

    return data



