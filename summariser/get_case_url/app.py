# summariser/get_case_url/app.py
import json, os, boto3
from decimal import Decimal
from botocore.exceptions import ClientError

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")
TABLE = dynamodb.Table(os.environ["SUMMARY_JOB_TABLE"])
BUCKET = os.environ["SUMMARY_BUCKET"]

def decimal_to_num(obj):
    """Convert Decimal objects to int/float for JSON serialization"""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_num(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_num(i) for i in obj]
    return obj

def lambda_handler(event, context):
    try:
        meeting_id = (event.get("queryStringParameters") or {}).get("meetingId")
        if not meeting_id:
            return _r(400, {"error": "Missing meetingId in query params"})

        # Get the job row
        res = TABLE.get_item(Key={"meetingId": meeting_id})
        item = res.get("Item")
        if not item:
            return _r(404, {"error": "Job not found", "meetingId": meeting_id})

        # Check case check workflow status (not summary status)
        case_check_status = (item.get("caseCheckStatus") or "UNKNOWN").upper()
        if case_check_status not in ["COMPLETED", "IN_REVIEW"]:
            return _r(200, {
                "meetingId": meeting_id,
                "caseCheckStatus": case_check_status,
                "message": f"Case check status: {case_check_status}"
            })

        # Get the caseCheckKey from DynamoDB (set by the workflow)
        case_key = item.get("caseCheckKey")

        if not case_key:
            return _r(404, {
                "error": "Case check key not found in DynamoDB",
                "meetingId": meeting_id,
                "caseCheckStatus": case_check_status,
                "hint": "The workflow may not have completed successfully"
            })

        # Verify the file exists in S3
        try:
            s3.head_object(Bucket=BUCKET, Key=case_key)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return _r(404, {
                    "error": "Case check file not found in S3",
                    "meetingId": meeting_id,
                    "caseCheckStatus": case_check_status,
                    "caseCheckKey": case_key,
                    "hint": "DynamoDB has the key but S3 file is missing"
                })
            raise  # Re-raise other errors

        # Fetch the actual case check data from S3
        try:
            response = s3.get_object(Bucket=BUCKET, Key=case_key)
            case_check_data = json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            return _r(500, {
                "error": "Failed to read case check data from S3",
                "details": str(e),
                "caseCheckKey": case_key
            })

        # Return the case check data along with metadata
        response = {
            "meetingId": meeting_id,
            "caseCheckStatus": case_check_status,
            "caseCheckKey": case_key,
            "casePassRate": decimal_to_num(item.get("casePassRate")),
            "data": case_check_data
        }

        # Add A2I info if in review
        if case_check_status == "IN_REVIEW":
            a2i_loop = item.get("a2iCaseLoop")
            if a2i_loop:
                response["a2iCaseLoop"] = a2i_loop

        return _r(200, response)

    except Exception as e:
        return _r(500, {"error": str(e)})

def _r(c,b): return {"statusCode": c, "headers":{"Content-Type":"application/json"}, "body": json.dumps(b)}
