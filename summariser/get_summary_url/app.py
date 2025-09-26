import json
import boto3
import os
from botocore.exceptions import ClientError

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

SUMMARY_BUCKET = os.environ["SUMMARY_BUCKET"]
JOB_TABLE = os.environ["SUMMARY_JOB_TABLE"]

TABLE = dynamodb.Table(JOB_TABLE)

def lambda_handler(event, context):
    try:
        meeting_id = (event.get("queryStringParameters") or {}).get("meetingId")
        if not meeting_id:
            return _resp(400, {"error": "Missing meetingId in query params"})

        # Get the job row
        res = TABLE.get_item(Key={"meetingId": meeting_id})
        item = res.get("Item")
        if not item:
            return _resp(404, {"error": "Job not found", "meetingId": meeting_id})

        status = (item.get("status") or "UNKNOWN").upper()
        if status != "COMPLETED":
            # pass back status only while processing
            return _resp(200, {"meetingId": meeting_id, "status": status})

        # Use the exact key written by the worker (dated prefix + versioned filename)
        key = item.get("summaryKey")
        latest_key = item.get("latestSummaryKey")

        # Try to find a valid key (latest first, then fallback to original)
        valid_key = None

        # First try the latest key (new format)
        if latest_key:
            try:
                s3.head_object(Bucket=SUMMARY_BUCKET, Key=latest_key)
                valid_key = latest_key
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise  # Re-raise if not a 404

        # Fallback to summaryKey (legacy format)
        if not valid_key and key:
            try:
                s3.head_object(Bucket=SUMMARY_BUCKET, Key=key)
                valid_key = key
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise  # Re-raise if not a 404

        # Migration fallback: check for files with previous schema versions
        checked_keys = [latest_key, key]
        if not valid_key:
            from utils.s3_partitioner import get_s3_partitioner
            from datetime import datetime, timezone
            import re

            # Try to extract date from meeting ID or fall back to current date
            now = datetime.now(timezone.utc)
            year, month = now.year, now.month

            # Check for files that should exist in new partitioned format based on old schema versions
            partitioner = get_s3_partitioner()

            # Try different schema versions and time periods
            schema_versions = ["1.2", "1.1", "1.0"]
            time_periods = [
                (year, month),
                (year, (month-1) if month > 1 else 12),  # Previous month
                (year-1 if month == 1 else year, 12 if month == 1 else month-1)  # Handle year boundary
            ]

            migration_keys = []

            # Check new partitioned format for different versions
            for version in schema_versions:
                for check_year, check_month in time_periods:
                    new_format_key = f"summaries/version={version}/year={check_year}/month={check_month:02d}/meeting_id={meeting_id}/summary.json"
                    migration_keys.append(new_format_key)

            # Check legacy format paths
            for check_year, check_month in time_periods:
                for version in ["v1.1", "v1.0", ""]:
                    version_suffix = f".{version}" if version else ""
                    legacy_format_key = f"summaries/{check_year}/{check_month:02d}/{meeting_id}/summary{version_suffix}.json"
                    migration_keys.append(legacy_format_key)

            # Check all migration paths
            for migration_key in migration_keys:
                if migration_key not in checked_keys:  # Avoid duplicate checks
                    try:
                        s3.head_object(Bucket=SUMMARY_BUCKET, Key=migration_key)
                        valid_key = migration_key
                        checked_keys.append(migration_key)
                        break
                    except ClientError as e:
                        if e.response['Error']['Code'] != '404':
                            raise  # Re-raise if not a 404
                        checked_keys.append(migration_key)

        if not valid_key:
            return _resp(404, {"error": "Summary file not found in S3", "meetingId": meeting_id, "checkedKeys": checked_keys})

        # Sign it
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": SUMMARY_BUCKET, "Key": valid_key},
            ExpiresIn=300  # 5 minutes
        )

        return _resp(200, {
            "meetingId": meeting_id,
            "status": "COMPLETED",
            "downloadUrl": url
        })

    except Exception as e:
        return _resp(500, {"error": str(e)})

def _resp(code, body):
    return {
        "statusCode": code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body)
    }
