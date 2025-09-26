import json
import re
import random
import math
import time
from datetime import datetime, timezone
from botocore.exceptions import ClientError
import boto3
from constants import AWS_REGION

# AWS client
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def strip_code_fences(s: str) -> str:
    # remove ```json ... ``` or ``` ... ```
    s = s.strip()
    fence = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S | re.I)
    m = fence.match(s)
    return m.group(1) if m else s

def extract_first_json(s: str) -> str:
    """
    Return the first balanced {...} JSON object found in s.
    Handles leading prose and trailing notes.
    """
    s = strip_code_fences(s)
    start = s.find("{")
    if start == -1:
        return s.strip()
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1].strip()
    # if unbalanced, fallback to original
    return s[start:].strip()

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

def bedrock_infer(model_id: str, body: str, tries: int = 5, base: float = 0.6, max_sleep: float = 6.0):
    """
    Invoke Bedrock with exponential backoff + jitter.
    Returns parsed JSON response body (dict).
    """
    last_err = None
    for attempt in range(1, tries + 1):
        t0 = time.perf_counter()
        try:
            resp = bedrock.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            raw = resp["body"].read().decode("utf-8")
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return raw, latency_ms
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

