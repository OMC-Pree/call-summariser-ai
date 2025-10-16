# summariser/list_summaries/app.py
import json
import os
import boto3
from decimal import Decimal

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

SUMMARY_BUCKET = os.environ["SUMMARY_BUCKET"]
JOB_TABLE = os.environ["SUMMARY_JOB_TABLE"]
TABLE = dynamodb.Table(JOB_TABLE)
PREFIX_VERSION = "summaries/v=1.2"  # Only support v1.2
PROJECTION = (
    "meetingId, #s, updatedAt, summaryKey, caseCheckKey, "
    "sentiment, actionCount, casePassRate, caseFailedCount, "
    "promptVersion, modelVersion, a2iCaseLoop, employerName"
)

# ---------- Helpers ----------
def _to_jsonable(obj):
    """Recursively convert Decimals -> int/float so json.dumps won't choke."""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj

def _resp(code, body):
    return {
        "statusCode": code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(_to_jsonable(body)),
    }

def _presign_or_none(key: str | None, expires=300) -> str | None:
    if not key:
        return None
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": SUMMARY_BUCKET, "Key": key},
        ExpiresIn=expires,
    )

# ---------- Handler ----------
def lambda_handler(event, _context):
    qs = event.get("queryStringParameters") or {}

    # Pagination cap (soft)
    try:
        limit = int(qs.get("limit", "200"))
    except (ValueError, TypeError):
        limit = 200
    limit = max(1, min(limit, 1000))

    # Status slice (default COMPLETED)
    status_filter = (qs.get("status") or "COMPLETED").upper()

    # Build a minimal projection
    ean = {"#s": "status"}

    # Full scan (table is small). Server-side DDB FilterExpression would still read items,
    # so we filter in-code and stop once we hit 'limit'.
    items = []
    resp = TABLE.scan(ProjectionExpression=PROJECTION, ExpressionAttributeNames=ean)
    while True:
        for it in resp.get("Items", []):
            if (str(it.get("status") or "")).upper() != status_filter:
                continue

            items.append(it)
            if len(items) >= limit:
                break

        if len(items) >= limit or "LastEvaluatedKey" not in resp:
            break

        resp = TABLE.scan(
            ExclusiveStartKey=resp["LastEvaluatedKey"],
            ProjectionExpression=PROJECTION,
            ExpressionAttributeNames=ean,
        )

    # Sort newest first and build response rows with presigned URLs
    items.sort(key=lambda x: str(x.get("updatedAt") or ""), reverse=True)
    items = items[:limit]

    out = []
    for it in items:
        summary_key = it.get("summaryKey")
        case_key = it.get("caseCheckKey")

        out.append({
            "meetingId": it.get("meetingId"),
            "updatedAt": it.get("updatedAt"),
            "status": it.get("status"),
            "a2iCaseLoop": it.get("a2iCaseLoop"),
            "sentiment": it.get("sentiment"),
            "actionCount": it.get("actionCount"),
            "casePassRate": it.get("casePassRate"),
            "caseFailedCount": it.get("caseFailedCount"),
            "promptVersion": it.get("promptVersion"),
            "modelVersion": it.get("modelVersion"),
            "employerName": it.get("employerName") or "No employer",
            "prefixVersion": PREFIX_VERSION,
            "summaryUrl": _presign_or_none(summary_key),
            "caseUrl": _presign_or_none(case_key),
        })

    return _resp(200, {"items": out})
