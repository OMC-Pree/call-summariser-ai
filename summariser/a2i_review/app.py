"""
A2I Review Lambda - Step Functions workflow step
Initiates SageMaker A2I human review loop for failed cases
"""
import json
import time
import re
import boto3
from botocore.exceptions import ClientError
from utils import helper
from utils.error_handler import lambda_error_handler, ValidationError
from constants import *

a2i = boto3.client("sagemaker-a2i-runtime", region_name=AWS_REGION)
s3 = boto3.client("s3")


def get_transcript_from_s3(s3_key: str) -> str:
    """Fetch transcript from S3"""
    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=s3_key)
    return response['Body'].read().decode('utf-8')


def _safe_loop_name(prefix: str, meeting_id: str) -> str:
    """
    Produce a SageMaker A2I-safe loop name (alnum + hyphen, <= 63 chars).
    """
    base = re.sub(r'[^a-zA-Z0-9-]', '-', f"{prefix}-{meeting_id}")[:40]
    return f"{base}-{int(time.time())}"[:63]


@lambda_error_handler()
def lambda_handler(event, context):
    """
    Start A2I human review loop for case check failures.

    Input:
        - meetingId: str
        - caseCheckKey: str (S3 key to case check JSON)
        - redactedTranscriptKey: str (S3 key to redacted transcript)

    Output:
        - humanLoopName: str
        - status: str (started|disabled|error)
    """
    meeting_id = event.get("meetingId")
    case_check_key = event.get("caseCheckKey")
    transcript_key = event.get("redactedTranscriptKey")

    if not meeting_id:
        raise ValidationError("meetingId is required")

    if not case_check_key:
        raise ValidationError("caseCheckKey is required")

    # Fetch case data from S3
    try:
        case_response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=case_check_key)
        case_data = json.loads(case_response['Body'].read().decode('utf-8'))
    except Exception as e:
        helper.log_json("ERROR", "A2I_CASE_DATA_FETCH_FAILED", meetingId=meeting_id, error=str(e)[:200])
        raise ValidationError(f"Failed to fetch case data from S3: {e}")

    # Fetch transcript excerpt from S3 (first 4000 chars for A2I)
    redacted_transcript = ""
    if transcript_key:
        try:
            full_transcript = get_transcript_from_s3(transcript_key)
            redacted_transcript = full_transcript[:4000]
        except Exception as e:
            helper.log_json("WARN", "A2I_TRANSCRIPT_FETCH_FAILED", meetingId=meeting_id, error=str(e)[:200])

    # Check if A2I is enabled
    if not A2I_FLOW_ARN_CASE:
        helper.log_json("INFO", "A2I_DISABLED", meetingId=meeting_id)
        return {
            "humanLoopName": None,
            "status": "disabled"
        }

    try:
        # Create human loop
        loop_name = _safe_loop_name("case", meeting_id)

        overall = case_data.get("overall", {}) or {}
        pass_rate = float(overall.get("pass_rate", 0.0))

        input_content = {
            "meeting_id": meeting_id,
            "pass_rate": pass_rate,
            "transcript_excerpt": (redacted_transcript or "")[:4000],
            "case_json": json.dumps(case_data)[:4000]
        }

        a2i.start_human_loop(
            HumanLoopName=loop_name,
            FlowDefinitionArn=A2I_FLOW_ARN_CASE,
            HumanLoopInput={"InputContent": json.dumps(input_content)},
            DataAttributes={"ContentClassifiers": ["FreeOfPersonallyIdentifiableInformation"]}
        )

        helper.log_json("INFO", "A2I_LOOP_STARTED", meetingId=meeting_id, humanLoopName=loop_name)

        return {
            "humanLoopName": loop_name,
            "status": "started"
        }

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        helper.log_json("ERROR", "A2I_LOOP_FAILED", meetingId=meeting_id, errorCode=error_code, error=str(e)[:500])

        return {
            "humanLoopName": None,
            "status": "error"
        }
    except Exception as e:
        helper.log_json("ERROR", "A2I_UNEXPECTED_ERROR", meetingId=meeting_id, error=str(e)[:500])

        return {
            "humanLoopName": None,
            "status": "error"
        }
