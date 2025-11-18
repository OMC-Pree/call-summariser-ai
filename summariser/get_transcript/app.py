"""
Get Redacted Transcript Lambda - API Gateway endpoint
Returns the PII-redacted transcript for a meeting
"""
import json
import boto3
import os
from utils import helper
from utils.error_handler import lambda_error_handler, ValidationError
from constants import SUMMARY_BUCKET, S3_PREFIX, SCHEMA_VERSION

s3 = boto3.client("s3")


def find_redacted_transcript(meeting_id: str) -> tuple[str, str]:
    """
    Find the redacted transcript for a meeting by searching S3.
    Returns (s3_key, transcript_content) or raises ValidationError if not found.
    """
    # Try to find the redacted transcript in S3
    # It could be in different locations depending on when it was processed

    # Pattern 1: Athena-partitioned path (current)
    # summaries/supplementary/version=1.2/year=YYYY/month=MM/meeting_id=XXX/redacted_transcript.txt
    prefix = f"{S3_PREFIX}/supplementary/"

    try:
        helper.log_json("INFO", "SEARCHING_REDACTED_TRANSCRIPT",
                       meetingId=meeting_id, prefix=prefix)

        # List all objects with the meeting_id in the path
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=SUMMARY_BUCKET,
            Prefix=prefix
        )

        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                # Check if this key contains our meeting_id and is a redacted transcript
                if f"meeting_id={meeting_id}" in key and key.endswith("redacted_transcript.txt"):
                    helper.log_json("INFO", "FOUND_REDACTED_TRANSCRIPT",
                                   meetingId=meeting_id, s3Key=key)

                    # Fetch the transcript content
                    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=key)
                    content = response['Body'].read().decode('utf-8')
                    return key, content

        # If not found in supplementary, try legacy location
        # {meeting_id}/redacted_transcript.txt or cleaned-transcript.txt
        legacy_keys = [
            f"{meeting_id}/redacted_transcript.txt",
            f"{meeting_id}/cleaned-transcript.txt"
        ]

        for legacy_key in legacy_keys:
            try:
                helper.log_json("INFO", "TRYING_LEGACY_PATH",
                               meetingId=meeting_id, s3Key=legacy_key)
                response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=legacy_key)
                content = response['Body'].read().decode('utf-8')
                helper.log_json("INFO", "FOUND_LEGACY_TRANSCRIPT",
                               meetingId=meeting_id, s3Key=legacy_key)
                return legacy_key, content
            except s3.exceptions.NoSuchKey:
                continue

        raise ValidationError(f"Redacted transcript not found for meeting {meeting_id}")

    except ValidationError:
        raise
    except Exception as e:
        helper.log_json("ERROR", "TRANSCRIPT_SEARCH_FAILED",
                       meetingId=meeting_id, error=str(e))
        raise ValidationError(f"Failed to retrieve transcript: {str(e)}")


@lambda_error_handler()
def lambda_handler(event, context):
    """
    GET /transcript/{meetingId}

    Returns the PII-redacted transcript for a meeting.

    Response:
        200: { meeting_id: str, transcript: str, s3_key: str }
        404: { error: "Transcript not found" }
        500: { error: "Internal error message" }
    """
    # CORS headers for S3-hosted frontend
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET,OPTIONS"
    }

    # Handle OPTIONS preflight request
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": cors_headers,
            "body": ""
        }

    # Extract meeting_id from path parameters
    path_params = event.get("pathParameters", {})
    meeting_id = path_params.get("meetingId")

    if not meeting_id:
        return {
            "statusCode": 400,
            "headers": cors_headers,
            "body": json.dumps({"error": "meetingId is required in path"})
        }

    helper.log_json("INFO", "GET_TRANSCRIPT_REQUEST", meetingId=meeting_id)

    try:
        # Find and retrieve the redacted transcript
        s3_key, transcript = find_redacted_transcript(meeting_id)

        return {
            "statusCode": 200,
            "headers": {
                **cors_headers,
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "meeting_id": meeting_id,
                "transcript": transcript,
                "s3_key": s3_key,
                "type": "redacted"  # Always return redacted version
            })
        }

    except ValidationError as e:
        helper.log_json("WARN", "TRANSCRIPT_NOT_FOUND",
                       meetingId=meeting_id, error=str(e))
        return {
            "statusCode": 404,
            "headers": cors_headers,
            "body": json.dumps({"error": str(e)})
        }
    except Exception as e:
        helper.log_json("ERROR", "GET_TRANSCRIPT_FAILED",
                       meetingId=meeting_id, error=str(e))
        return {
            "statusCode": 500,
            "headers": cors_headers,
            "body": json.dumps({"error": "Failed to retrieve transcript"})
        }
