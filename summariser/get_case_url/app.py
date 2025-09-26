# summariser/get_case_url/app.py
import json, os, boto3
from botocore.exceptions import ClientError

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")
TABLE = dynamodb.Table(os.environ["SUMMARY_JOB_TABLE"])
BUCKET = os.environ["SUMMARY_BUCKET"]

def lambda_handler(event, context):
    meeting_id = (event.get("queryStringParameters") or {}).get("meetingId")
    if not meeting_id:
        return _r(400, {"error": "Missing meetingId"})

    # Search for case check files in the correct A2I output location
    case_check_prefix = "a2i/outputs/case-check-flow-min/"

    try:
        # List all files in the case check directory for this meeting
        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=case_check_prefix
        )

        if 'Contents' not in response:
            return _r(404, {"error": "No case check files found", "meetingId": meeting_id})

        # Find files that contain this meeting ID
        matching_files = []
        for obj in response['Contents']:
            key = obj['Key']
            if meeting_id in key and key.endswith('output.json'):
                matching_files.append({
                    'key': key,
                    'last_modified': obj['LastModified']
                })

        if not matching_files:
            return _r(404, {"error": "No case check files found for this meeting", "meetingId": meeting_id})

        # Use the most recent case check file
        matching_files.sort(key=lambda x: x['last_modified'], reverse=True)
        latest_file = matching_files[0]

        # Verify the file exists and generate presigned URL
        s3.head_object(Bucket=BUCKET, Key=latest_file['key'])
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": latest_file['key']},
            ExpiresIn=300
        )

        return _r(200, {
            "meetingId": meeting_id,
            "caseUrl": url,
            "caseCheckKey": latest_file['key']
        })

    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return _r(404, {"error": "Case check file not found in S3", "meetingId": meeting_id})
        raise
    except Exception as e:
        return _r(500, {"error": f"Failed to retrieve case check file: {str(e)}", "meetingId": meeting_id})

def _r(c,b): return {"statusCode": c, "headers":{"Content-Type":"application/json"}, "body": json.dumps(b)}
