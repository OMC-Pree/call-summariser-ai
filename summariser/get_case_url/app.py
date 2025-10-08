# summariser/get_case_url/app.py
import json, os, boto3
from botocore.exceptions import ClientError

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")
TABLE = dynamodb.Table(os.environ["SUMMARY_JOB_TABLE"])
BUCKET = os.environ["SUMMARY_BUCKET"]

def lambda_handler(event, context):
    try:
        meeting_id = (event.get("queryStringParameters") or {}).get("meetingId")
        if not meeting_id:
            return _r(400, {"error": "Missing meetingId in query params"})

        # Get the job row (same as summary function)
        res = TABLE.get_item(Key={"meetingId": meeting_id})
        item = res.get("Item")
        if not item:
            return _r(404, {"error": "Job not found", "meetingId": meeting_id})

        status = (item.get("status") or "UNKNOWN").upper()
        if status != "COMPLETED":
            return _r(200, {"meetingId": meeting_id, "status": status})

        # Try to find case check file using the same pattern as summary
        # Look for caseCheckKey in DynamoDB first
        case_key = item.get("caseCheckKey")

        valid_key = None
        checked_keys = []

        # Try the caseCheckKey from DynamoDB
        if case_key:
            try:
                s3.head_object(Bucket=BUCKET, Key=case_key)
                valid_key = case_key
                checked_keys.append(case_key)
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise  # Re-raise if not a 404
                checked_keys.append(case_key)

        # Only check for current v1.2 schema if not found in DynamoDB
        if not valid_key:
            from datetime import datetime, timezone

            # Use current date for partitioned path
            now = datetime.now(timezone.utc)
            year, month = now.year, now.month

            # Check v1.2 partitioned format only
            case_key_v12 = f"summaries/version=1.2/year={year}/month={month:02d}/meeting_id={meeting_id}/case_check.v1.0.json"
            try:
                s3.head_object(Bucket=BUCKET, Key=case_key_v12)
                valid_key = case_key_v12
                checked_keys.append(case_key_v12)
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise  # Re-raise if not a 404
                checked_keys.append(case_key_v12)

        if not valid_key:
            return _r(404, {
                "error": "Case check file not found in S3",
                "meetingId": meeting_id,
                "checkedKeys": checked_keys,
                "totalKeysChecked": len(checked_keys),
                "dbItem": {
                    "status": status,
                    "hasCaseCheckKey": case_key is not None,
                    "caseCheckKey": case_key
                }
            })

        # Generate presigned URL
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": BUCKET, "Key": valid_key},
            ExpiresIn=300  # 5 minutes
        )

        return _r(200, {
            "meetingId": meeting_id,
            "caseUrl": url,
            "caseCheckKey": valid_key
        })

    except Exception as e:
        return _r(500, {"error": str(e)})

def _r(c,b): return {"statusCode": c, "headers":{"Content-Type":"application/json"}, "body": json.dumps(b)}
