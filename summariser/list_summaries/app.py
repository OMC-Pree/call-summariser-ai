# summariser/list_summaries/app.py
import json
import os
import re
import boto3
from decimal import Decimal

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

SUMMARY_BUCKET     = os.environ["SUMMARY_BUCKET"]
JOB_TABLE          = os.environ["SUMMARY_JOB_TABLE"]
S3_PREFIX_CURRENT  = os.getenv("S3_PREFIX", "summaries")  # e.g. "summaries/v=1.1" or just "summaries"
TABLE = dynamodb.Table(JOB_TABLE)

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

def _norm_prefix(p: str) -> str:
    """Normalize 'summaries/v=1.1/' -> 'summaries/v=1.1' (no trailing slash)."""
    return (p or "").strip().strip("/")

_V_RE = re.compile(r"^v=.+$")

def _infer_prefix_version_from_key(key: str) -> str | None:
    """
    Infer the logical prefix "summaries" or "summaries/v=..." from an S3 key path.
    Examples:
      summaries/2025/09/om-123/summary.v1.1.json        -> 'summaries'
      summaries/v=1.1/2025/09/om-123/summary.v1.1.json  -> 'summaries/v=1.1'
    """
    if not key:
        return None
    parts = key.split("/")
    if not parts:
        return None
    root = parts[0]
    if len(parts) > 1 and _V_RE.match(parts[1]):
        return f"{root}/{parts[1]}"
    return root

def _presign_or_none(key: str | None, expires=300) -> str | None:
    if not key:
        return None
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": SUMMARY_BUCKET, "Key": key},
        ExpiresIn=expires,
    )

# ---------- Handler ----------
def lambda_handler(event, context):
    qs = event.get("queryStringParameters") or {}

    # Pagination cap (soft)
    try:
        limit = int(qs.get("limit", "200"))
    except Exception:
        limit = 200
    limit = max(1, min(limit, 1000))

    # Status slice (default COMPLETED)
    status_filter = (qs.get("status") or "COMPLETED").upper()

    # Version slice. If omitted, default to current S3_PREFIX. 'all'/* disables version filtering.
    requested_version = _norm_prefix(qs.get("version") or S3_PREFIX_CURRENT)
    version_filter_enabled = requested_version not in ("", "all", "*")

    # Build a minimal projection (add employerName/prefixVersion for FE)
    proj = (
        "meetingId, #s, updatedAt, summaryKey, caseCheckKey, "
        "sentiment, actionCount, casePassRate, caseFailedCount, "
        "promptVersion, modelVersion, a2iCaseLoop, employerName, prefixVersion"
    )
    ean = {"#s": "status"}

    # Full scan (table is small). Server-side DDB FilterExpression would still read items,
    # so we filter in-code and stop once we hit 'limit'.
    items = []
    resp = TABLE.scan(ProjectionExpression=proj, ExpressionAttributeNames=ean)
    while True:
        for it in resp.get("Items", []):
            if (str(it.get("status") or "")).upper() != status_filter:
                continue

            # Version filter
            if version_filter_enabled:
                pv = _norm_prefix(it.get("prefixVersion") or "")
                if pv:
                    if pv != requested_version:
                        continue
                else:
                    # Back-compat: infer from summaryKey if prefixVersion missing
                    inferred = _infer_prefix_version_from_key(it.get("summaryKey") or "")
                    if _norm_prefix(inferred or "") != requested_version:
                        continue

            items.append(it)
            if len(items) >= limit:
                break

        if len(items) >= limit or "LastEvaluatedKey" not in resp:
            break

        resp = TABLE.scan(
            ExclusiveStartKey=resp["LastEvaluatedKey"],
            ProjectionExpression=proj,
            ExpressionAttributeNames=ean,
        )

    # Sort newest first and build response rows with presigned URLs
    items.sort(key=lambda x: str(x.get("updatedAt") or ""), reverse=True)
    items = items[:limit]

    out = []
    for it in items:
        summary_key = it.get("summaryKey")
        case_key = it.get("caseCheckKey")

        # Choose the best-known prefixVersion for the FE
        pv = it.get("prefixVersion") or _infer_prefix_version_from_key(summary_key) or requested_version

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
            "prefixVersion": pv,
            "summaryUrl": _presign_or_none(summary_key),
            "caseUrl": _presign_or_none(case_key),
        })

    return _resp(200, {"items": out})
