"""
Case Check Lambda - Step Functions workflow step
Performs compliance case checking using Bedrock Claude
"""
import json
import boto3
import os
from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from utils import helper
from utils.error_handler import lambda_error_handler
from utils.prompt_management import invoke_with_prompt_management
from constants import *

# Import KB retrieval module
try:
    from case_check.kb_retrieval import retrieve_and_format_examples
    KB_ENABLED = True
except ImportError:
    KB_ENABLED = False
    helper.log_json("WARNING", "KB_MODULE_NOT_FOUND", message="Running without Knowledge Base integration")

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
s3 = boto3.client("s3")
ssm = boto3.client("ssm", region_name=AWS_REGION)

# Prompt Management configuration
USE_PROMPT_MANAGEMENT = os.environ.get("USE_PROMPT_MANAGEMENT", "true").lower() == "true"
PROMPT_PARAM_NAME = os.environ.get("PROMPT_PARAM_NAME_CASE_CHECK", "/call-summariser/prompts/case-check/current")

# Cache for prompt ARN (loaded once per Lambda container)
_cached_prompt_arn = None

# Get KB configuration from environment
USE_KB = os.environ.get("USE_KNOWLEDGE_BASE", "true").lower() == "true"
KB_PARAM_NAME = os.environ.get("KNOWLEDGE_BASE_PARAM_NAME", "/call-summariser/knowledge-base-id")

# Fetch KB ID from Parameter Store (cached for Lambda container lifetime)
KB_ID = None
if USE_KB:
    try:
        response = ssm.get_parameter(Name=KB_PARAM_NAME, WithDecryption=False)
        KB_ID = response['Parameter']['Value']
        helper.log_json("INFO", "KB_ID_LOADED_FROM_PARAMETER_STORE", kb_param_name=KB_PARAM_NAME)
    except Exception as e:
        helper.log_json("WARNING", "KB_ID_PARAMETER_NOT_FOUND",
                       kb_param_name=KB_PARAM_NAME,
                       error=str(e),
                       message="Knowledge Base integration will be disabled")


def get_prompt_arn() -> Optional[str]:
    """
    Get prompt ARN from Parameter Store (cached for Lambda container lifetime).
    """
    global _cached_prompt_arn

    if not USE_PROMPT_MANAGEMENT:
        return None

    if _cached_prompt_arn is not None:
        return _cached_prompt_arn

    try:
        response = ssm.get_parameter(Name=PROMPT_PARAM_NAME)
        _cached_prompt_arn = response['Parameter']['Value']
        helper.log_json("INFO", "CASE_CHECK_PROMPT_ARN_LOADED",
                       parameter_name=PROMPT_PARAM_NAME,
                       prompt_arn=_cached_prompt_arn)
        return _cached_prompt_arn
    except Exception as e:
        helper.log_json("WARNING", "CASE_CHECK_PROMPT_ARN_LOAD_FAILED",
                       parameter_name=PROMPT_PARAM_NAME,
                       error=str(e))
        return None


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
        # - Then Competent (fully compliant)
        # - Then CompetentWithDevelopment (compliant but could improve)
        # - Then Inconclusive
        # - Finally NotApplicable
        # Within same status, use highest confidence
        priority = {"Fail": 5, "Competent": 4, "CompetentWithDevelopment": 3, "Inconclusive": 2, "NotApplicable": 1}
        best_check = max(check_across_chunks, key=lambda c: (priority.get(c["status"], 0), c.get("confidence", 0)))

        # Combine evidence quotes from all chunks that found evidence
        all_quotes = [c.get("evidence_quote", "") for c in check_across_chunks
                     if c.get("evidence_quote") and c["status"] in ["Competent", "CompetentWithDevelopment", "Fail"]]
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
    competent_checks = sum(1 for r in merged["results"] if r["status"] in ["Competent", "CompetentWithDevelopment"])
    pass_rate = competent_checks / total_checks if total_checks > 0 else 0

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
Status = Literal["Competent", "CompetentWithDevelopment", "Fail", "NotApplicable", "Inconclusive"]


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


# ---------- Tool definition for structured output ----------
def get_case_check_tool():
    """
    Create a tool definition for structured case check output.
    This ensures the LLM returns valid JSON matching our Pydantic schema.
    """
    return {
        "toolSpec": {
            "name": "submit_case_check",
            "description": "Submit the case check assessment results. Focus on providing the 'results' array with your assessment of each check, and the 'overall' summary. Metadata fields are optional and will be populated automatically.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "check_schema_version": {"type": "string"},
                        "session_type": {"type": "string"},
                        "checklist_version": {"type": "string"},
                        "meeting_id": {"type": "string"},
                        "model_version": {"type": "string"},
                        "prompt_version": {"type": "string"},
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Check identifier"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["Competent", "CompetentWithDevelopment", "Fail", "NotApplicable", "Inconclusive"],
                                        "description": "Assessment status"
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                        "description": "Confidence score 0-1"
                                    },
                                    "evidence_spans": {
                                        "type": "array",
                                        "items": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                            "minItems": 2,
                                            "maxItems": 2
                                        },
                                        "description": "List of [start, end] character positions"
                                    },
                                    "evidence_quote": {
                                        "type": "string",
                                        "description": "REQUIRED: Direct quote from transcript supporting your assessment. Must be actual dialogue from the call."
                                    },
                                    "comment": {
                                        "type": "string",
                                        "description": "REQUIRED: Brief explanation of your assessment (1-2 sentences)"
                                    }
                                },
                                "required": ["id", "status", "confidence", "evidence_quote", "evidence_spans", "comment"]
                            }
                        },
                        "overall": {
                            "type": "object",
                            "properties": {
                                "pass_rate": {"type": "number"},
                                "failed_ids": {"type": "array", "items": {"type": "string"}},
                                "high_severity_flags": {"type": "array", "items": {"type": "string"}},
                                "has_high_severity_failures": {"type": "boolean"}
                            }
                        }
                    },
                    "required": ["results", "overall"]
                }
            }
        }
    }


STARTER_SESSION_CHECKS = [
    # Compliance Criteria
    {"id": "call_recording_confirmed", "prompt": "Call recording confirmed? Did the coach confirm that the call is being recorded for training and compliance purposes?", "required": True, "severity": "high"},
    {"id": "regulated_advice_given", "prompt": "Was regulated financial advice given and/or was there evidence of steering/social norming? NOTE: This is NOT permitted. Regulated advice means specific product recommendations or steering towards specific actions. If the coach gave regulated advice or steered the client, the status must be 'Fail'.", "required": True, "severity": "high"},
    {"id": "vulnerability_identified", "prompt": "Was any vulnerability identified and addressed appropriately? Did the coach identify any client vulnerabilities (financial, health, life circumstances) and handle them appropriately?", "required": True, "severity": "high"},
    {"id": "dob_confirmed", "prompt": "Date of Birth confirmed? Was the client's date of birth confirmed during the call?", "required": True, "severity": "medium"},
    {"id": "client_name_confirmed", "prompt": "Client name confirmed? Was the client's full name confirmed during the call?", "required": True, "severity": "medium"},
    {"id": "marital_status_confirmed", "prompt": "Client's marital status confirmed? Was the client's marital/partner status confirmed?", "required": True, "severity": "medium"},
    {"id": "citizenship_confirmed", "prompt": "UK Citizenship and if any US tax connections confirmed? Did the coach confirm UK citizenship/residency and check for any US tax connections?", "required": True, "severity": "medium"},
    {"id": "dependents_confirmed", "prompt": "Dependents confirmed? Were dependents confirmed? Note: Dependents are not limited to just children.", "required": True, "severity": "medium"},
    {"id": "pension_details_confirmed", "prompt": "Pension details confirmed? Did the coach confirm the client's pension details (current pensions, contributions, amounts)?", "required": True, "severity": "medium"},
    {"id": "income_expenditure_confirmed", "prompt": "Income and expenditure details confirmed? Did the coach confirm the client's income and expenditure details?", "required": True, "severity": "medium"},
    {"id": "assets_liabilities_confirmed", "prompt": "Assets and liabilities details confirmed? Did the coach confirm the client's assets (savings, property, investments) and liabilities (debts, loans)?", "required": True, "severity": "medium"},
    {"id": "emergency_fund_confirmed", "prompt": "Emergency fund confirmed? Did the coach discuss and confirm the client's emergency fund status?", "required": True, "severity": "medium"},
    {"id": "will_confirmed", "prompt": "Will confirmed? Did the coach confirm whether the client has a will in place?", "required": True, "severity": "low"},
    {"id": "pension_withdrawal_if_over_50", "prompt": "If over 50, will the client be withdrawing from their pension within the next 5 years? If the client is over 50, did the coach check if they plan to withdraw from their pension in the next 5 years?", "required": False, "severity": "medium"},
    {"id": "high_interest_debt_addressed", "prompt": "If the client has high-interest unsecured debt, did the coach let them know they won't be able to produce any recommendations until that debt is paid off?", "required": False, "severity": "high"},
    {"id": "fees_charges_explained", "prompt": "Were fees and charges correctly explained to the client? Did the coach clearly explain the service fees (e.g., £299, salary sacrifice options)?", "required": True, "severity": "high"},
    {"id": "way_forward_agreed", "prompt": "Was a way forward agreed with the client? Did the coach and client agree on next steps and book a follow-up session?", "required": True, "severity": "medium"},

    # Macro-Criteria (Coaching Quality) - COMMENTED OUT to reduce token requirements
    # These can be assessed manually. Keeping only critical compliance checks for automated checking.
    # {"id": "coach_introduction_signposting", "prompt": "Did the coach introduce themselves and Octopus Money, and signpost the structure of this call? Did the coach provide a clear introduction and outline of what the call would cover?", "required": True, "severity": "medium"},
    # {"id": "client_goals_established", "prompt": "Did the coach establish key information about the client's goals? Did the coach ask about and explore the client's financial goals?", "required": True, "severity": "high"},
    # {"id": "current_actions_established", "prompt": "Did the coach establish what the client is already doing to work towards their goals? Did the coach explore existing actions, savings, investments, or plans?", "required": True, "severity": "medium"},
    # {"id": "client_motivations_established", "prompt": "Did the coach establish client motivations for achieving their goals? Did the coach explore WHY the goals are important to the client?", "required": True, "severity": "medium"},
    # {"id": "relevant_suggestions_provided", "prompt": "Were relevant suggestions provided to the client based on the goals explored? Did the coach provide practical, relevant suggestions tailored to the client's specific goals and circumstances?", "required": True, "severity": "high"},
    # {"id": "money_calculators_introduced", "prompt": "Did the coach introduce the money calculators? Did the coach explain and introduce the money calculators that the client will use?", "required": True, "severity": "medium"},
    # {"id": "asked_client_move_forward", "prompt": "Did the coach clearly ask the client if they want to move forward with the service? Did the coach explicitly ask if the client wants to sign up and continue?", "required": True, "severity": "high"},
    # {"id": "client_questions_opportunity", "prompt": "Did the client have the opportunity to ask any questions? Did the coach provide opportunities throughout and at the end for the client to ask questions?", "required": True, "severity": "medium"}
]


# JSON repair and extraction functions removed - no longer needed with structured output via Tool Use


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
    # Reduced chunk size to ensure output fits within Claude 3 Sonnet's 4096 token limit
    CHUNK_SIZE = 20000  # ~5000 tokens per chunk (smaller chunks = less detailed output = fits in 4K limit)
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

    # Retrieve KB examples once (reuse across all chunks)
    kb_examples = ""
    if USE_KB and KB_ENABLED and KB_ID:
        try:
            import time
            kb_start_time = time.time()
            helper.log_json("INFO", "KB_RETRIEVAL_START", meetingId=meeting_id, kb_id=KB_ID)

            # Get check IDs and descriptions
            check_ids = [c["id"] for c in STARTER_SESSION_CHECKS]
            check_descriptions = {c["id"]: c["prompt"] for c in STARTER_SESSION_CHECKS}

            kb_examples = retrieve_and_format_examples(
                check_ids=check_ids,
                check_descriptions=check_descriptions,
                max_per_check=1,
                kb_id=KB_ID
            )

            kb_elapsed = time.time() - kb_start_time
            helper.log_json("INFO", "KB_RETRIEVAL_COMPLETE",
                          meetingId=meeting_id,
                          examples_length=len(kb_examples),
                          kb_retrieval_time_ms=int(kb_elapsed * 1000))
        except Exception as e:
            helper.log_json("WARNING", "KB_RETRIEVAL_ERROR",
                          meetingId=meeting_id,
                          error=str(e),
                          message="Continuing without KB examples")
            kb_examples = ""
    else:
        helper.log_json("INFO", "KB_DISABLED",
                       meetingId=meeting_id,
                       use_kb=USE_KB,
                       kb_enabled=KB_ENABLED,
                       has_kb_id=bool(KB_ID))

    # Process each chunk
    chunk_results = []
    truncated_chunks = []
    for idx, chunk in enumerate(chunks):
        helper.log_json("INFO", f"PROCESSING_CHUNK_{idx + 1}_OF_{num_chunks}",
                       meetingId=meeting_id,
                       chunk_length=len(chunk))

        # Get tool definition for guaranteed JSON schema
        case_check_tool = get_case_check_tool()

        # Get prompt ARN
        prompt_arn = get_prompt_arn()

        if prompt_arn:
            # Use Prompt Management
            checklist_json = json.dumps(
                {"session_type": "starter_session", "version": "1", "checks": STARTER_SESSION_CHECKS},
                ensure_ascii=False,
            )

            variables = {
                "kb_examples": kb_examples if kb_examples else "",
                "checklist_json": checklist_json,
                "cleaned_transcript": chunk
            }

            resp, latency_ms = invoke_with_prompt_management(
                prompt_arn=prompt_arn,
                variables=variables,
                model_id=MODEL_ID,
                tools=[case_check_tool],
                tool_choice={"tool": {"name": "submit_case_check"}},
                max_tokens_override=4000
            )
        else:
            # Fallback to inline prompt (should not happen with USE_PROMPT_MANAGEMENT=true)
            helper.log_json("ERROR", "NO_PROMPT_ARN", message="Prompt Management disabled or failed")
            raise ValueError("Prompt Management is required but prompt ARN not available")

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
            raise ValueError(f"No tool use block found in response for chunk {idx + 1}")

        # Get validated JSON from tool input
        validated_json = tool_use_block["input"]

        # Inject metadata fields that LLM shouldn't generate
        # The LLM should only focus on the actual case check results
        validated_json.setdefault("check_schema_version", CASE_CHECK_SCHEMA_VERSION)
        validated_json.setdefault("session_type", "starter_session")
        validated_json.setdefault("checklist_version", "1")
        validated_json.setdefault("meeting_id", meeting_id)
        validated_json.setdefault("model_version", MODEL_VERSION)
        validated_json.setdefault("prompt_version", PROMPT_VERSION)

        # Defensive parsing: Claude 3 Sonnet sometimes returns arrays as strings
        validated_json = helper.parse_stringified_fields(
            data=validated_json,
            fields=["results"],
            meeting_id=meeting_id,
            context=f"case_check_chunk_{idx + 1}"
        )

        # Calculate evidence_spans if missing or empty
        if "results" in validated_json and isinstance(validated_json["results"], list):
            for result_item in validated_json["results"]:
                if isinstance(result_item, dict):
                    evidence_quote = result_item.get("evidence_quote", "")
                    evidence_spans = result_item.get("evidence_spans", [])

                    # If evidence_spans is empty but we have evidence_quote, calculate it
                    if evidence_quote and not evidence_spans:
                        # Find the quote in the chunk
                        # Clean up the quote for matching
                        clean_quote = evidence_quote.strip()
                        if clean_quote:
                            pos = chunk.find(clean_quote)
                            if pos != -1:
                                result_item["evidence_spans"] = [[pos, pos + len(clean_quote)]]
                            else:
                                # Try fuzzy match with first 50 chars
                                short_quote = clean_quote[:50]
                                pos = chunk.find(short_quote)
                                if pos != -1:
                                    result_item["evidence_spans"] = [[pos, pos + len(clean_quote)]]
                                else:
                                    # No match found, set empty span
                                    result_item["evidence_spans"] = []
                                    helper.log_json("WARNING", "EVIDENCE_QUOTE_NOT_FOUND_IN_CHUNK",
                                                   meetingId=meeting_id,
                                                   check_id=result_item.get("id"),
                                                   quote_preview=clean_quote[:100])

        # Check if response was truncated
        stop_reason = resp.get("stopReason")
        if stop_reason == "max_tokens":
            truncated_chunks.append(idx + 1)
            helper.log_json("WARNING", "CHUNK_CASECHECK_TRUNCATED",
                           meetingId=meeting_id,
                           chunk_idx=idx + 1,
                           message="Chunk response hit max_tokens - skipping chunk to avoid partial/malformed data")
            # Skip this truncated chunk - partial compliance data is unreliable
            continue

        usage = resp.get("usage", {})

        # Log with cache metrics for cost tracking
        log_data = {
            "meetingId": meeting_id,
            "latency_ms": latency_ms,
            "stop_reason": stop_reason,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "structured_output": True
        }

        # Add cache metrics if available (only present when using prompt caching)
        if "cacheReadInputTokens" in usage:
            log_data["cache_read_tokens"] = usage.get("cacheReadInputTokens", 0)
            log_data["cache_creation_tokens"] = usage.get("cacheCreationInputTokens", 0)
            # Calculate cost savings (cache reads are 90% cheaper)
            cache_savings = usage.get("cacheReadInputTokens", 0) * 0.9
            log_data["estimated_cache_savings_tokens"] = int(cache_savings)

        helper.log_json("INFO", f"CHUNK_{idx + 1}_LLM_OK", **log_data)

        # Validate with Pydantic (should always succeed with tool use)
        parsed = CaseCheckPayload.model_validate(validated_json)
        chunk_results.append(parsed.model_dump())

    # Check if we have any successful chunks
    if not chunk_results:
        raise ValueError(f"No chunks processed successfully for meeting {meeting_id}. "
                        f"Total chunks: {num_chunks}, Truncated/failed: {len(truncated_chunks)}")

    # Merge results from all chunks
    if len(chunk_results) > 1:
        helper.log_json("INFO", "MERGING_CHUNK_RESULTS",
                       meetingId=meeting_id,
                       total_chunks=num_chunks,
                       successful_chunks=len(chunk_results),
                       truncated_chunks=len(truncated_chunks))
        data = merge_case_check_results(chunk_results)
    else:
        data = chunk_results[0]

    # Normalize data
    data["meeting_id"] = meeting_id
    data["model_version"] = MODEL_VERSION
    data["prompt_version"] = PROMPT_VERSION

    # Add truncation metadata if any chunks were truncated
    if truncated_chunks:
        if "overall" not in data:
            data["overall"] = {}
        data["overall"]["truncated_chunks"] = truncated_chunks
        data["overall"]["has_truncation"] = True
        helper.log_json("WARNING", "CASE_CHECK_HAD_TRUNCATION",
                       meetingId=meeting_id,
                       truncated_chunks=truncated_chunks,
                       message="Some chunks were truncated - results may be incomplete")

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
