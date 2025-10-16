# summariser/initiate_summary/app.py
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
sqs = boto3.client("sqs")

# DynamoDB with retry wrapper
dynamodb_wrapper = DynamoDBRetryWrapper(SUMMARY_JOB_TABLE)

@lambda_error_handler()
def lambda_handler(event, context):
    logger.info("Processing initiate summary request")

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
        ["meetingId", "coachName"],
        "request body"
    )

    # Validate and sanitize inputs
    meeting_id = InputValidator.validate_meeting_id(body.get("meetingId"))
    coach_name = InputValidator.validate_string_field(
        body.get("coachName"), "coachName", min_length=2, max_length=100
    )

    # Optional employer name
    employer_name = ""
    if body.get("employerName"):
        employer_name = InputValidator.validate_string_field(
            body.get("employerName"), "employerName", min_length=1, max_length=100
        )

    # Validate transcript or zoom meeting ID (at least one required)
    transcript = InputValidator.sanitize_text(body.get("transcript", ""))
    zoom_meeting_id = str(body.get("zoomMeetingId") or "").replace(" ", "").strip()

    if not transcript and not zoom_meeting_id:
        raise ValidationError(
            "Either 'transcript' or 'zoomMeetingId' must be provided",
            details={"provided_fields": list(body.keys())}
        )

    # Validate zoom meeting ID format if provided
    if zoom_meeting_id and not zoom_meeting_id.isdigit():
        raise ValidationError(
            "zoomMeetingId must contain only digits",
            field="zoomMeetingId"
        )

    # Extract case checking option from request (optional)
    enable_case_check = bool(body.get("enableCaseCheck", False))

    # Extract force reprocess option from request (optional)
    force_reprocess = bool(body.get("forceReprocess", False))

    # 1) Fast path: job already completed? (skip if force reprocess enabled)
    if not force_reprocess and _job_completed(meeting_id):
        logger.info(f"Summary already exists for meeting {meeting_id}")
        return _response(200, {"message": "Summary already exists", "meetingId": meeting_id})

    if force_reprocess:
        logger.info(f"Force reprocess enabled for meeting {meeting_id}")

    # 2) Mark QUEUED without clobbering COMPLETED
    _mark_queued(meeting_id)

    # 3) Push to SQS
    _push_to_queue(meeting_id, coach_name, employer_name, transcript, zoom_meeting_id, enable_case_check, force_reprocess)

    logger.info(f"Job queued successfully for meeting {meeting_id}")
    return _response(202, {"message": "Job queued successfully", "meetingId": meeting_id})

# ---------- Helpers ----------

def _job_completed(meeting_id: str) -> bool:
    """
    Check if job is already completed using DynamoDB with S3 fallback.
    """
    try:
        # Try DynamoDB first (fast path)
        res = dynamodb_wrapper.get_item(Key={"meetingId": meeting_id})
        item = res.get("Item") or {}
        return (item.get("status") or "").upper() == "COMPLETED"
    except Exception as e:
        logger.warning(f"DynamoDB check failed for {meeting_id}, trying S3 fallback: {e}")

        # If DDB is unavailable, fall back to S3 best-effort (SAM local skips)
        if os.environ.get("AWS_SAM_LOCAL") == "true":
            return False

        return _check_s3_completion(meeting_id)

@with_s3_retry()
def _check_s3_completion(meeting_id: str) -> bool:
    """Check S3 for existing summary (fallback method)"""
    try:
        prefix = f"{S3_PREFIX}/"
        schema = SCHEMA_VERSION
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=SUMMARY_BUCKET, Prefix=prefix, MaxKeys=1000):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(f"/{meeting_id}/summary.v{schema}.json"):
                    return True
        return False
    except Exception as e:
        handle_s3_error(e, SUMMARY_BUCKET, correlation_id=meeting_id)
        return False

def _mark_queued(meeting_id: str) -> None:
    """Mark meeting as queued in DynamoDB"""
    item = {
        "meetingId": meeting_id,
        "status": "QUEUED",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }

    if os.environ.get("AWS_SAM_LOCAL") == "true":
        logger.info(f"🧪 [Mock] Would set QUEUED for {meeting_id}")
        return

    try:
        dynamodb_wrapper.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(#s) OR #s <> :done",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":done": "COMPLETED"},
        )
        logger.info(f"Marked meeting {meeting_id} as QUEUED")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ConditionalCheckFailedException":
            # Someone already completed it between our check and this write
            logger.info(f"Meeting {meeting_id} already completed, skipping queue")
        else:
            # Re-raise other DynamoDB errors
            raise ExternalServiceError(
                f"Failed to mark meeting as queued: {e}",
                service="dynamodb",
                correlation_id=meeting_id
            )

def _push_to_queue(meeting_id: str, coach_name: str, employer_name: str, transcript: str, zoom_meeting_id: str, enable_case_check: bool = False, force_reprocess: bool = False) -> None:
    """Push job message to SQS queue"""
    msg = {
        "meetingId": meeting_id,
        "coachName": coach_name,
        "employerName": employer_name,
        "enableCaseCheck": enable_case_check,
        "forceReprocess": force_reprocess
    }

    if transcript:
        msg["transcript"] = transcript
    else:
        msg["zoomMeetingId"] = zoom_meeting_id

    if os.environ.get("AWS_SAM_LOCAL") == "true":
        logger.info("🧪 [Mock] Would push message to SQS queue")
        logger.debug(f"Message content: {json.dumps(msg, indent=2)}")
        return

    try:
        response = sqs.send_message(
            QueueUrl=os.environ["SUMMARY_JOBS_QUEUE"],
            MessageBody=json.dumps(msg),
            MessageAttributes={
                'MeetingId': {
                    'StringValue': meeting_id,
                    'DataType': 'String'
                },
                'MessageType': {
                    'StringValue': 'SummaryRequest',
                    'DataType': 'String'
                }
            }
        )
        logger.info(f"Message sent to SQS for meeting {meeting_id}, MessageId: {response.get('MessageId')}")

    except Exception as e:
        raise ExternalServiceError(
            f"Failed to send message to SQS: {e}",
            service="sqs",
            correlation_id=meeting_id
        )

def _response(status_code: int, body: dict):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
