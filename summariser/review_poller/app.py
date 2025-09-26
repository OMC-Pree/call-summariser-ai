# summariser/review_poller/app.py
import os
import re
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

import boto3
from botocore.exceptions import ClientError
from constants import *

REGION = AWS_REGION
JOB_TABLE = SUMMARY_JOB_TABLE
S3_BUCKET = SUMMARY_BUCKET  # not strictly required here, but handy if you later write merged files

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(JOB_TABLE)
a2i = boto3.client("sagemaker-a2i-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    s3://bucket/prefix/... -> (bucket, key)
    """
    m = re.match(r"^s3://([^/]+)/(.+)$", uri)
    if not m:
        raise ValueError(f"Bad S3 URI: {uri}")
    return m.group(1), m.group(2)


def _scan_in_review_jobs() -> list[Dict[str, Any]]:
    """
    Returns all items with status = IN_REVIEW and an a2iCaseLoop value.
    Uses a Scan with a FilterExpression for simplicity (small volumes). Scale later with a GSI if needed.
    """
    from boto3.dynamodb.conditions import Attr

    items: list[Dict[str, Any]] = []
    kwargs = {
        "FilterExpression": Attr("status").eq("IN_REVIEW") & Attr("a2iCaseLoop").exists(),
        "ProjectionExpression": "meetingId, #s, a2iCaseLoop, updatedAt",
        "ExpressionAttributeNames": {"#s": "status"},
        "Limit": 100,
    }
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


def _describe_loop(loop_name: str) -> Dict[str, Any]:
    return a2i.describe_human_loop(HumanLoopName=loop_name)


def _fetch_human_output_json(info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Reads the human loop output JSON from S3 if status == Completed.
    Returns parsed dict or None if not completed.
    """
    if info.get("HumanLoopStatus") != "Completed":
        return None
    out = info.get("HumanLoopOutput") or {}
    uri = out.get("OutputS3Uri")
    if not uri:
        return {"_warning": "No OutputS3Uri in DescribeHumanLoop response"}
    bkt, key = _parse_s3_uri(uri)
    obj = s3.get_object(Bucket=bkt, Key=key)
    body = obj["Body"].read()
    return json.loads(body)


def _extract_reviewer_fields(human_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    A2I answer payload shape:
      {
        "humanAnswers": [
          {
            "answerContent": {
              "overall_decision": "approve|edit|reject",
              "corrected_json": "...",
              "reviewer_comments": "..."
            },
            ...
          }
        ],
        "humanLoopName": "...",
        "creationTime": "...",
        ...
      }
    """
    ans_list = human_json.get("humanAnswers") or []
    ans = (ans_list[0] or {}) if ans_list else {}
    content = ans.get("answerContent") or {}
    decision = content.get("overall_decision", "approve")
    corrected = content.get("corrected_json")
    comments = content.get("reviewer_comments", "")
    return {
        "decision": decision,
        "corrected_json": corrected,
        "comments": comments,
    }


def _complete_item(meeting_id: str, extra: Dict[str, Any]) -> None:
    """
    Overwrite item status to COMPLETED with extra metadata.
    We intentionally PUT the full object (simple pattern used elsewhere in your code),
    but here we only update keys we care about to avoid clobbering.
    """
    # Safer: UpdateExpression to only set fields we add/touch
    update_expr = (
        "SET #s = :status, "
        "updatedAt = :ts, "
        "humanReviewed = :hr, "
        "humanReview = :hrd"
    )
    expr_names = {"#s": "status"}
    expr_vals = {
        ":status": "COMPLETED",
        ":ts": _now_iso(),
        ":hr": True,
        ":hrd": {
            "source": "A2I",
            "decision": extra.get("decision"),
            "comments": extra.get("comments"),
            "loopName": extra.get("loopName"),
            "outputUri": extra.get("outputUri"),
        },
    }
    # Optionally carry forward casePassRate/fails if provided (harmless if missing)
    for k in ("casePassRate", "caseFailedCount"):
        if k in extra:
            update_expr += f", {k} = :{k}"
            expr_vals[f":{k}"] = extra[k]

    table.update_item(
        Key={"meetingId": meeting_id},
        UpdateExpression=update_expr,
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_vals,
    )


def _mark_failed_review(meeting_id: str, loop_name: str, status: str) -> None:
    """
    If a loop ended in Failed/Stopped/Expired, we still move the job on to COMPLETED
    (no human corrections), but record the review outcome.
    """
    table.update_item(
        Key={"meetingId": meeting_id},
        UpdateExpression=(
            "SET #s = :status, updatedAt = :ts, "
            "humanReviewed = :hr, humanReview = :hrd"
        ),
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":status": "COMPLETED",
            ":ts": _now_iso(),
            ":hr": False,
            ":hrd": {
                "source": "A2I",
                "decision": f"review_{status.lower()}",
                "loopName": loop_name,
            },
        },
    )


def lambda_handler(event, context):
    processed = 0
    try:
        items = _scan_in_review_jobs()
        if not items:
            return {"ok": True, "processed": 0, "note": "no IN_REVIEW items"}

        for it in items:
            meeting_id = it.get("meetingId")
            loop_name = it.get("a2iCaseLoop")
            if not meeting_id or not loop_name:
                continue

            try:
                info = _describe_loop(loop_name)
                hl_status = info.get("HumanLoopStatus")
                if hl_status == "InProgress":
                    continue  # still with reviewer

                if hl_status == "Completed":
                    out_json = _fetch_human_output_json(info) or {}
                    fields = _extract_reviewer_fields(out_json)
                    # Keep a pointer to the output object for audits
                    output_uri = (info.get("HumanLoopOutput") or {}).get("OutputS3Uri")
                    fields.update({
                        "loopName": loop_name,
                        "outputUri": output_uri,
                    })
                    _complete_item(meeting_id, fields)
                    processed += 1
                    continue

                # Other terminal states: Failed | Stopped | Expired
                if hl_status in ("Failed", "Stopped", "Expired"):
                    _mark_failed_review(meeting_id, loop_name, hl_status)
                    processed += 1
                    continue

                # Unexpected state: log and continue
                print(f"[REVIEW] {meeting_id} loop {loop_name} unexpected status: {hl_status}")

            except ClientError as e:
                print(f"[REVIEW] Error describing loop {loop_name} for {meeting_id}: {e}")

        return {"ok": True, "processed": processed}
    except Exception as e:
        print(f"[REVIEW] Fatal error: {e}")
        return {"ok": False, "error": str(e), "processed": processed}
