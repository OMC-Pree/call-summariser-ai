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

        # Only check v1.2 format if not found in DynamoDB
        checked_keys = [latest_key, key]
        if not valid_key:
            from datetime import datetime, timezone

            # Use current date for partitioned path
            now = datetime.now(timezone.utc)
            year, month = now.year, now.month

            # Check v1.2 partitioned format only
            v12_key = f"summaries/version=1.2/year={year}/month={month:02d}/meeting_id={meeting_id}/summary.json"
            try:
                s3.head_object(Bucket=SUMMARY_BUCKET, Key=v12_key)
                valid_key = v12_key
                checked_keys.append(v12_key)
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise  # Re-raise if not a 404
                checked_keys.append(v12_key)

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
