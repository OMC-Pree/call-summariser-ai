import json
import random
import time
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

# strip_code_fences and extract_first_json removed - no longer needed with structured output

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
        system: Optional system prompt string
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
                request_params["system"] = [{"text": system}]

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



