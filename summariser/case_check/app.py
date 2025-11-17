"""
Case Check Lambda - Step Functions workflow step
Performs compliance case checking using Bedrock Claude
"""
import json
import os
from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from utils import helper
from utils.error_handler import lambda_error_handler
from utils.prompt_management import invoke_with_prompt_management, get_prompt_arn_from_parameter_store
from utils.aws_clients import AWSClients
from constants import *

# Import KB retrieval module
try:
    from case_check.kb_retrieval import retrieve_and_format_examples
    KB_ENABLED = True
except ImportError:
    KB_ENABLED = False
    helper.log_json("WARNING", "KB_MODULE_NOT_FOUND", message="Running without Knowledge Base integration")

# Use centralized AWS clients
bedrock = AWSClients.bedrock_runtime()
s3 = AWSClients.s3()
ssm = AWSClients.ssm()

# Prompt Management configuration
USE_PROMPT_MANAGEMENT = os.environ.get("USE_PROMPT_MANAGEMENT", "true").lower() == "true"
PROMPT_PARAM_NAME = os.environ.get("PROMPT_PARAM_NAME_CASE_CHECK", "/call-summariser/prompts/case-check/current")

# Cache for prompt ARN (loaded once per Lambda container)
_prompt_arn_cache = {}

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
    """Get prompt ARN from Parameter Store (cached for Lambda container lifetime)"""
    return get_prompt_arn_from_parameter_store(
        param_name=PROMPT_PARAM_NAME,
        cache_dict=_prompt_arn_cache,
        use_prompt_management=USE_PROMPT_MANAGEMENT
    )


def get_transcript_from_s3(s3_key: str) -> str:
    """Fetch transcript from S3"""
    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=s3_key)
    return response['Body'].read().decode('utf-8')


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
    {"id": "vulnerability_identified", "prompt": "Was any vulnerability identified and addressed appropriately? Did the coach identify any client vulnerabilities according to FCA FG21/1 guidelines and handle them appropriately?\n\nVulnerability Categories (FCA FG21/1):\n- Health: Mental Health Condition, Severe or Long-term Illness, Hearing or Visual Impairment, Physical Disability, Addiction, Low Mental Capacity\n- Life Events: Bereavement, Caring Responsibilities, Domestic Abuse, Relationship Breakdown, Income Shock, Retirement\n- Resilience: Low Emotional Resilience, Inadequate or Erratic Income, Over-indebtedness, Low Savings\n- Capability: Low Knowledge or Confidence in Managing Finances, Poor Literacy or Numeracy Skills, Poor English Language Skills, Poor Digital Skills, Learning Difficulties, Low Access to Help or Support", "required": True, "severity": "high"},
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
        - forceReprocess: bool (optional, default False)

    Output:
        - caseData: dict
        - caseKey: str
        - passRate: float
    """
    transcript_key = event.get("redactedTranscriptKey")
    meeting_id = event.get("meetingId")
    force_reprocess = event.get("forceReprocess", False)

    if not transcript_key:
        raise ValueError("redactedTranscriptKey is required")

    if not meeting_id:
        raise ValueError("meetingId is required")

    # Determine expected case check S3 key
    now = datetime.now(timezone.utc)
    if ATHENA_PARTITIONED:
        case_key = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={now.year}/month={now.month:02d}/meeting_id={meeting_id}/case_check.v{CASE_CHECK_SCHEMA_VERSION}.json"
    else:
        case_key = f"{S3_PREFIX}/{now.year:04d}/{now.month:02d}/{meeting_id}/case_check.v{CASE_CHECK_SCHEMA_VERSION}.json"

    # Idempotency check: If case check already exists and not forcing reprocess, return it
    if not force_reprocess:
        try:
            response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=case_key)
            case_data = json.loads(response['Body'].read().decode('utf-8'))
            pass_rate = float(case_data.get("overall", {}).get("pass_rate", 0.0))
            helper.log_json("INFO", "CASE_CHECK_EXISTS", meetingId=meeting_id, caseKey=case_key, reused=True)
            return {
                "caseData": case_data,
                "caseKey": case_key,
                "passRate": pass_rate
            }
        except s3.exceptions.NoSuchKey:
            # File doesn't exist, proceed with case check
            pass
        except Exception as e:
            # Log error but continue with processing
            helper.log_json("WARN", "CASE_CHECK_CHECK_FAILED", meetingId=meeting_id, error=str(e))

    # Fetch transcript from S3
    full_transcript = get_transcript_from_s3(transcript_key)

    # Claude 3.7 Sonnet context window: 200K tokens (sufficient for max 1.5h calls)
    # Max call: 1.5h ≈ 90K chars ≈ 22.5K tokens input + 5K output = 27.5K total (13% of context)
    # No chunking needed - process entire transcript in one API call for better context
    helper.log_json("INFO", "CASE_CHECK_START",
                   meetingId=meeting_id,
                   transcript_length=len(full_transcript),
                   transcript_tokens_approx=len(full_transcript) // 4)

    # Retrieve KB examples for compliance checks
    kb_examples = ""
    if USE_KB and KB_ENABLED and KB_ID:
        try:
            import time
            kb_start_time = time.time()
            helper.log_json("INFO", "KB_RETRIEVAL_START", meetingId=meeting_id, kb_id=KB_ID)

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

    # Process full transcript in single API call
    case_check_tool = get_case_check_tool()
    prompt_arn = get_prompt_arn()

    if prompt_arn:
        checklist_json = json.dumps(
            {"session_type": "starter_session", "version": "1", "checks": STARTER_SESSION_CHECKS},
            ensure_ascii=False,
        )

        variables = {
            "kb_examples": kb_examples if kb_examples else "",
            "checklist_json": checklist_json,
            "cleaned_transcript": full_transcript
        }

        resp, latency_ms = invoke_with_prompt_management(
            prompt_arn=prompt_arn,
            variables=variables,
            model_id=MODEL_ID,
            tools=[case_check_tool],
            tool_choice={"tool": {"name": "submit_case_check"}},
            max_tokens_override=8000
        )
    else:
        helper.log_json("ERROR", "NO_PROMPT_ARN", message="Prompt Management disabled or failed")
        raise ValueError("Prompt Management is required but prompt ARN not available")

    # Extract structured output from tool use
    output_message = resp.get("output", {}).get("message", {})
    content_blocks = output_message.get("content", [])

    tool_use_block = None
    for block in content_blocks:
        if "toolUse" in block:
            tool_use_block = block["toolUse"]
            break

    if not tool_use_block:
        raise ValueError("No tool use block found in response")

    stop_reason = resp.get("stopReason", "")
    if stop_reason == "max_tokens":
        helper.log_json("ERROR", "CASE_CHECK_TRUNCATED",
                       meetingId=meeting_id,
                       message="Response hit max_tokens - increase max_tokens_override or reduce transcript length")
        raise ValueError(f"Response truncated at max_tokens for meeting {meeting_id}")

    validated_json = tool_use_block["input"]

    # Inject metadata fields
    validated_json.setdefault("check_schema_version", CASE_CHECK_SCHEMA_VERSION)
    validated_json.setdefault("session_type", "starter_session")
    validated_json.setdefault("checklist_version", "1")
    validated_json.setdefault("meeting_id", meeting_id)
    validated_json.setdefault("model_version", MODEL_VERSION)
    validated_json.setdefault("prompt_version", PROMPT_VERSION)

    # Defensive parsing for stringified fields (only if needed)
    # Tool Use should return structured JSON, but handle edge cases
    if "results" in validated_json and isinstance(validated_json["results"], str):
        helper.log_json("WARNING", "RESULTS_AS_STRING",
                       meetingId=meeting_id,
                       message="results field unexpectedly returned as string, attempting to parse",
                       stopReason=stop_reason,
                       resultsLength=len(validated_json["results"]))

        try:
            validated_json = helper.parse_stringified_fields(
                data=validated_json,
                fields=["results"],
                meeting_id=meeting_id,
                context="case_check"
            )
        except ValueError as e:
            # If parsing fails, log safe metadata without PII
            helper.log_json("ERROR", "RESULTS_PARSE_FAILED",
                           meetingId=meeting_id,
                           parseError=str(e),
                           stopReason=stop_reason,
                           resultsLength=len(validated_json["results"]),
                           resultsType=str(type(validated_json["results"])),
                           message="Failed to parse results field - may be truncated or malformed")
            raise ValueError(f"Failed to parse case check results for meeting {meeting_id}. "
                           f"Stop reason: {stop_reason}. Error: {str(e)}") from e

    # Calculate evidence_spans if missing
    if "results" in validated_json and isinstance(validated_json["results"], list):
        for result_item in validated_json["results"]:
            if isinstance(result_item, dict):
                evidence_quote = result_item.get("evidence_quote", "")
                evidence_spans = result_item.get("evidence_spans", [])

                if evidence_quote and not evidence_spans:
                    clean_quote = evidence_quote.strip()
                    if clean_quote:
                        pos = full_transcript.find(clean_quote)
                        if pos != -1:
                            result_item["evidence_spans"] = [[pos, pos + len(clean_quote)]]
                        else:
                            short_quote = clean_quote[:50]
                            pos = full_transcript.find(short_quote)
                            if pos != -1:
                                result_item["evidence_spans"] = [[pos, pos + len(clean_quote)]]
                            else:
                                result_item["evidence_spans"] = []
                                helper.log_json("WARNING", "EVIDENCE_QUOTE_NOT_FOUND",
                                               meetingId=meeting_id,
                                               check_id=result_item.get("id"),
                                               quote_preview=clean_quote[:100])

    usage = resp.get("usage", {})
    cost_breakdown = helper.calculate_bedrock_cost(usage, model_id="claude-3-7-sonnet")

    log_data = {
        "meetingId": meeting_id,
        "operation": "case_check",
        "latency_ms": latency_ms,
        "stop_reason": stop_reason,
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "structured_output": True,
        "cost_usd": cost_breakdown["total_cost"],
        "input_cost_usd": cost_breakdown["input_cost"],
        "output_cost_usd": cost_breakdown["output_cost"]
    }

    if "cacheReadInputTokens" in usage:
        log_data["cache_read_tokens"] = usage.get("cacheReadInputTokens", 0)
        log_data["cache_creation_tokens"] = usage.get("cacheCreationInputTokens", 0)
        log_data["cache_read_cost_usd"] = cost_breakdown["cache_read_cost"]
        log_data["cache_write_cost_usd"] = cost_breakdown["cache_write_cost"]
        log_data["cache_savings_usd"] = cost_breakdown["cache_savings"]

    helper.log_json("INFO", "CASE_CHECK_LLM_OK", **log_data)

    # Validate with Pydantic
    parsed = CaseCheckPayload.model_validate(validated_json)
    data = parsed.model_dump()

    data["meeting_id"] = meeting_id
    data["model_version"] = MODEL_VERSION
    data["prompt_version"] = PROMPT_VERSION

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
