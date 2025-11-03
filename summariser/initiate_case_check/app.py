# summariser/initiate_case_check/app.py
import json
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timezone
import os
import logging


# Import error handling and retry mechanisms
from utils.error_handler import (
    lambda_error_handler, InputValidator, ValidationError,
    ExternalServiceError, handle_s3_error
)
from utils.retry_handler import DynamoDBRetryWrapper, with_s3_retry
from constants import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS clients with retry wrappers
s3 = boto3.client("s3")
sfn = boto3.client("stepfunctions")

# DynamoDB with retry wrapper
dynamodb_wrapper = DynamoDBRetryWrapper(SUMMARY_JOB_TABLE)

# Step Functions State Machine ARN
STATE_MACHINE_ARN = os.environ.get("CASE_CHECK_STATE_MACHINE_ARN", "")

@lambda_error_handler()
def lambda_handler(event, context):
    logger.info("Processing initiate case check request")

    # Parse body for API Gateway or direct invoke
    if "body" in event:
        body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
    else:
        body = event

    # Input validation
    if not isinstance(body, dict):
        raise ValidationError("Request body must be a JSON object")

    # Validate required fields
    InputValidator.validate_required_fields(
        body,
        ["meetingId"],
        "request body"
    )

    # Validate and sanitize inputs
    meeting_id = InputValidator.validate_meeting_id(body.get("meetingId"))

    # Optional coach name (needed if fetching transcript from Zoom)
    coach_name = body.get("coachName", "")
    if coach_name:
        coach_name = InputValidator.validate_string_field(
            coach_name, "coachName", min_length=2, max_length=100
        )

    # Check if redactedTranscriptKey is provided (skip transcript processing)
    redacted_transcript_key = body.get("redactedTranscriptKey", "")
    if redacted_transcript_key:
        # Validate S3 key format
        if not redacted_transcript_key.startswith(S3_PREFIX):
            raise ValidationError(
                f"redactedTranscriptKey must start with '{S3_PREFIX}'",
                field="redactedTranscriptKey"
            )
        logger.info(f"Using existing redacted transcript: {redacted_transcript_key}")
    else:
        # Validate transcript or zoom meeting ID (at least one required)
        transcript = InputValidator.sanitize_text(body.get("transcript", ""))
        zoom_meeting_id = str(body.get("zoomMeetingId") or "").replace(" ", "").strip()

        if not transcript and not zoom_meeting_id:
            raise ValidationError(
                "Either 'transcript', 'zoomMeetingId', or 'redactedTranscriptKey' must be provided",
                details={"provided_fields": list(body.keys())}
            )

        # Validate zoom meeting ID format if provided
        if zoom_meeting_id and not zoom_meeting_id.isdigit():
            raise ValidationError(
                "zoomMeetingId must contain only digits",
                field="zoomMeetingId"
            )

    # Extract force reprocess option from request (optional)
    force_reprocess = bool(body.get("forceReprocess", False))

    # 1) Fast path: case check already completed? Return existing data
    if not force_reprocess:
        existing_data = _get_existing_case_check(meeting_id)
        if existing_data:
            logger.info(f"Case check already exists for meeting {meeting_id}, returning existing data")

            # Fetch the full case check data from S3 if key is available
            case_check_full_data = None
            case_check_key = existing_data.get("caseCheckKey")
            if case_check_key and case_check_key != "found_in_s3":
                try:
                    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=case_check_key)
                    case_check_full_data = json.loads(response['Body'].read().decode('utf-8'))
                except Exception as e:
                    logger.warning(f"Failed to fetch case check data from S3: {e}")

            response_data = {
                "message": "Case check already completed",
                "meetingId": meeting_id,
                "caseCheckKey": case_check_key,
                "casePassRate": existing_data.get("casePassRate"),
                "caseCheckStatus": existing_data.get("caseCheckStatus"),
                "completed": True
            }

            if case_check_full_data:
                response_data["data"] = case_check_full_data

            return _response(200, response_data)

    if force_reprocess:
        logger.info(f"Force reprocess enabled for meeting {meeting_id}")

    # 2) Mark QUEUED (allow overwriting COMPLETED if force_reprocess)
    try:
        _mark_queued(meeting_id, force=force_reprocess)
    except ValidationError as e:
        # Race condition: case check completed between initial check and queue
        # Return existing data if available in error details
        if e.details and e.details.get("completed"):
            logger.info(f"Case check completed during processing, returning existing data")

            # Fetch full data from S3
            case_check_full_data = None
            case_check_key = e.details.get("caseCheckKey")
            if case_check_key:
                try:
                    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=case_check_key)
                    case_check_full_data = json.loads(response['Body'].read().decode('utf-8'))
                except Exception as fetch_error:
                    logger.warning(f"Failed to fetch case check data from S3: {fetch_error}")

            response_data = {
                "message": "Case check already completed",
                "meetingId": meeting_id,
                "caseCheckKey": case_check_key,
                "casePassRate": e.details.get("casePassRate"),
                "caseCheckStatus": e.details.get("caseCheckStatus"),
                "completed": True
            }

            if case_check_full_data:
                response_data["data"] = case_check_full_data

            return _response(200, response_data)
        # Otherwise re-raise
        raise

    # 3) Start Step Functions execution
    execution_arn = _start_step_function(
        meeting_id=meeting_id,
        coach_name=coach_name,
        transcript=body.get("transcript", ""),
        zoom_meeting_id=body.get("zoomMeetingId", ""),
        redacted_transcript_key=redacted_transcript_key,
        force_reprocess=force_reprocess
    )
    logger.info(f"Case check workflow started for meeting {meeting_id}: {execution_arn}")
    return _response(202, {"message": "Case check workflow started", "meetingId": meeting_id, "executionArn": execution_arn})

# ---------- Helpers ----------

def _get_existing_case_check(meeting_id: str) -> dict:
    """
    Get existing case check data if completed.
    Returns dict with case check data if exists and completed, otherwise None.
    """
    try:
        # Try DynamoDB first (fast path)
        res = dynamodb_wrapper.get_item(Key={"meetingId": meeting_id})
        item = res.get("Item") or {}

        # Check if case check workflow is completed (separate from summary workflow)
        case_check_status = (item.get("caseCheckStatus") or "").upper()
        has_case_check = bool(item.get("caseCheckKey"))

        if case_check_status == "COMPLETED" and has_case_check:
            return {
                "caseCheckStatus": case_check_status,
                "caseCheckKey": item.get("caseCheckKey"),
                "casePassRate": float(item.get("casePassRate", 0.0))
            }
        return None
    except Exception as e:
        logger.warning(f"DynamoDB check failed for {meeting_id}, trying S3 fallback: {e}")

        # If DDB is unavailable, fall back to S3 best-effort (SAM local skips)
        if os.environ.get("AWS_SAM_LOCAL") == "true":
            return None

        # Check S3 for case check file
        if _check_s3_case_check_completion(meeting_id):
            return {"caseCheckStatus": "COMPLETED", "caseCheckKey": "found_in_s3", "casePassRate": 0.0}
        return None

@with_s3_retry()
def _check_s3_case_check_completion(meeting_id: str) -> bool:
    """Check S3 for existing case check (fallback method)"""
    try:
        prefix = f"{S3_PREFIX}/"
        schema = CASE_CHECK_SCHEMA_VERSION
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=SUMMARY_BUCKET, Prefix=prefix, MaxKeys=1000):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(f"/{meeting_id}/case_check.v{schema}.json"):
                    return True
        return False
    except Exception as e:
        handle_s3_error(e, SUMMARY_BUCKET, correlation_id=meeting_id)
        return False

def _mark_queued(meeting_id: str, force: bool = False) -> None:
    """
    Mark meeting case check as queued in DynamoDB.
    Uses update_item to preserve other workflow data (like summary status).

    Args:
        meeting_id: The meeting identifier
        force: If True, overwrite even if caseCheckStatus is COMPLETED (for force reprocess)
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    if os.environ.get("AWS_SAM_LOCAL") == "true":
        logger.info(f"🧪 [Mock] Would set caseCheckStatus=QUEUED for {meeting_id} (force={force})")
        return

    try:
        if force:
            # Force reprocess: unconditionally overwrite the caseCheckStatus
            dynamodb_wrapper.update_item(
                Key={"meetingId": meeting_id},
                UpdateExpression="SET caseCheckStatus = :status, caseCheckUpdatedAt = :updated, createdAt = if_not_exists(createdAt, :created)",
                ExpressionAttributeValues={
                    ":status": "QUEUED",
                    ":updated": now_iso,
                    ":created": now_iso
                }
            )
            logger.info(f"Marked case check {meeting_id} as QUEUED (force overwrite)")
        else:
            # Normal path: don't clobber COMPLETED caseCheckStatus
            dynamodb_wrapper.update_item(
                Key={"meetingId": meeting_id},
                UpdateExpression="SET caseCheckStatus = :status, caseCheckUpdatedAt = :updated, createdAt = if_not_exists(createdAt, :created)",
                ConditionExpression="attribute_not_exists(caseCheckStatus) OR caseCheckStatus <> :done",
                ExpressionAttributeValues={
                    ":status": "QUEUED",
                    ":updated": now_iso,
                    ":created": now_iso,
                    ":done": "COMPLETED"
                }
            )
            logger.info(f"Marked case check {meeting_id} as QUEUED")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ConditionalCheckFailedException":
            # Case check completed between our check and this write (race condition)
            # Return the existing data instead of erroring
            logger.warning(f"Case check {meeting_id} completed during check, fetching existing data")
            existing_data = _get_existing_case_check(meeting_id)
            if existing_data:
                # Raise special exception with data that caller can catch
                raise ValidationError(
                    "Case check already completed",
                    details={
                        "completed": True,
                        "meetingId": meeting_id,
                        "caseCheckKey": existing_data.get("caseCheckKey"),
                        "casePassRate": existing_data.get("casePassRate"),
                        "caseCheckStatus": existing_data.get("caseCheckStatus")
                    }
                )
            else:
                # Shouldn't happen, but handle defensively
                raise ValidationError(
                    f"Case check already completed for meeting {meeting_id}. Use forceReprocess=true to re-run.",
                    field="forceReprocess"
                )
        else:
            # Re-raise other DynamoDB errors
            raise ExternalServiceError(
                f"Failed to mark case check as queued: {e}",
                service="dynamodb",
                correlation_id=meeting_id
            )

def _start_step_function(
    meeting_id: str,
    coach_name: str,
    transcript: str,
    zoom_meeting_id: str,
    redacted_transcript_key: str,
    force_reprocess: bool = False
) -> str:
    """Start Step Functions execution for case checking"""
    input_data = {
        "meetingId": meeting_id,
        "forceReprocess": force_reprocess
    }

    # If redacted transcript key is provided, use it directly
    if redacted_transcript_key:
        input_data["redactedTranscriptKey"] = redacted_transcript_key
    else:
        # Otherwise, provide transcript source information
        if coach_name:
            input_data["coachName"] = coach_name

        if transcript:
            input_data["transcript"] = transcript
        elif zoom_meeting_id:
            input_data["zoomMeetingId"] = zoom_meeting_id

    if os.environ.get("AWS_SAM_LOCAL") == "true":
        logger.info("🧪 [Mock] Would start Case Check Step Functions execution")
        logger.debug(f"Input data: {json.dumps(input_data, indent=2)}")
        return f"arn:aws:states:eu-west-2:000000000000:execution:case-checker-workflow:mock-{meeting_id}"

    try:
        response = sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=f"case-check-{meeting_id}-{int(datetime.now(timezone.utc).timestamp())}",
            input=json.dumps(input_data)
        )
        logger.info(f"Case check workflow started for meeting {meeting_id}: {response['executionArn']}")
        return response['executionArn']

    except Exception as e:
        raise ExternalServiceError(
            f"Failed to start case check workflow: {e}",
            service="stepfunctions",
            correlation_id=meeting_id
        )


def _response(status_code: int, body: dict):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
