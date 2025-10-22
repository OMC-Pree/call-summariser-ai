"""
Persist Summary Lambda - Step Functions workflow step
Persists summary JSON to S3 and updates metadata index
"""
import json
import boto3
from datetime import datetime, timezone
from utils import helper
from utils.error_handler import lambda_error_handler, ValidationError
from utils.s3_partitioner import get_s3_partitioner
from constants import *

s3 = boto3.client("s3")


def _calculate_quality_score(data: dict, actions: list, themes: list) -> float:
    """Calculate quality score (0.0-1.0) based on summary content"""
    score = 0.0

    # Summary quality (30%)
    summary = data.get("summary", "")
    if summary and len(summary.strip()) > 50:
        summary_score = min(1.0, len(summary.strip()) / 500.0)
        score += summary_score * 0.3

    # Action items (25%)
    if actions:
        action_score = min(1.0, len(actions) / 5.0)
        score += action_score * 0.25

    # Theme identification (20%)
    if themes:
        theme_score = min(1.0, len(themes) / 4.0)
        avg_confidence = sum(t.get("confidence", 0) for t in themes) / len(themes)
        theme_score += avg_confidence * 0.1
        score += min(1.0, theme_score) * 0.2

    # Key points (15%)
    key_points = data.get("key_points", [])
    if key_points:
        points_score = min(1.0, len(key_points) / 5.0)
        score += points_score * 0.15

    # Sentiment analysis (10%)
    sentiment = data.get("sentiment_analysis", {})
    if sentiment and sentiment.get("label", "").strip():
        score += 0.1

    return min(1.0, max(0.0, score))


def _build_summary_payload(meeting_id: str, coach_name: str, employer_name: str, source: str, data: dict) -> dict:
    """Build summary JSON payload with metadata"""
    now_iso = datetime.now(timezone.utc).isoformat() + "Z"

    sentiment_data = data.get("sentiment_analysis", {})
    sentiment_label = sentiment_data.get("label", "Neutral")
    if sentiment_label not in ("Positive", "Neutral", "Negative"):
        sentiment_label = "Neutral"

    sentiment_confidence = max(0.0, min(1.0, float(sentiment_data.get("confidence", 0.0))))

    actions = [
        {
            "id": f"A{i+1}",
            "text": a.get("description", ""),
        }
        for i, a in enumerate(data.get("action_items", []) or [])
    ]

    themes = [
        {
            "id": t.get("id", ""),
            "label": t.get("label", ""),
            "group": t.get("group") or "General",
            "confidence": max(0.0, min(1.0, float(t.get("confidence", 0.0)))),
            "evidence_quote": t.get("evidence_quote") or ""
        }
        for t in (data.get("themes", []) or [])
    ]

    quality_score = _calculate_quality_score(data, actions, themes)

    return {
        "summary_schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "meeting": {
            "id": meeting_id,
            "employerName": employer_name,
            "coach": coach_name,
            "createdAt": now_iso,
        },
        "sentiment": {
            "label": sentiment_label,
            "confidence": sentiment_confidence,
            "evidence_spans": [],
        },
        "themes": themes,
        "summary": data.get("summary", ""),
        "actions": actions,
        "call_metadata": {
            "source": source,
            "saved_at": now_iso,
            "insights_version": INSIGHTS_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model_version": MODEL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "prefix_version": S3_PREFIX,
        },
        "insights": {
            "action_count": len(actions),
            "theme_count": len(themes),
            "sentiment_label": sentiment_label,
            "is_escalation_candidate": (sentiment_label == "Negative"),
            "risk_flags": [],
            "categories": [],
        },
        "quality_score": quality_score,
    }


def _summary_object_keys(meeting_id: str) -> tuple:
    """Returns (run_key, latest_key) for summary JSON"""
    timestamp = datetime.now(timezone.utc)

    if ATHENA_PARTITIONED:
        run_key = f"{S3_PREFIX}/data/version={SCHEMA_VERSION}/year={timestamp.year}/month={timestamp.month:02d}/meeting_id={meeting_id}/summary_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
        latest_key = f"{S3_PREFIX}/data/version={SCHEMA_VERSION}/year={timestamp.year}/month={timestamp.month:02d}/meeting_id={meeting_id}/summary.json"
    else:
        run_id = f"schema={SCHEMA_VERSION}/model={MODEL_VERSION}/prompt={PROMPT_VERSION}/ts={timestamp.strftime('%Y%m%dT%H%M%SZ')}"
        run_key = f"{S3_PREFIX}/meetings/{meeting_id}/{run_id}/summary.json"
        latest_key = f"{S3_PREFIX}/meetings/{meeting_id}/latest/summary.json"

    return run_key, latest_key


def _save_summary_json(meeting_id: str, payload: dict, force: bool = False) -> tuple:
    """
    Save summary to S3 and return (run_key, latest_key).

    Args:
        meeting_id: The meeting identifier
        payload: The summary data to save
        force: If True, overwrite existing summary (for force reprocess)
    """
    run_key, latest_key = _summary_object_keys(meeting_id)

    # Check if summary already exists (skip check if force=True)
    if not force:
        try:
            s3.head_object(Bucket=SUMMARY_BUCKET, Key=latest_key)
            helper.log_json("INFO", "SUMMARY_EXISTS", meetingId=meeting_id)
            return run_key, latest_key
        except Exception:
            pass

    body = json.dumps(payload).encode("utf-8")

    # Save to latest key
    s3.put_object(
        Bucket=SUMMARY_BUCKET,
        Key=latest_key,
        Body=body,
        ContentType="application/json",
        CacheControl="no-store",
    )

    if force:
        helper.log_json("INFO", "SUMMARY_OVERWRITTEN", meetingId=meeting_id, key=latest_key, force=True)
    else:
        helper.log_json("INFO", "SUMMARY_SAVED", meetingId=meeting_id, key=latest_key)

    return latest_key, latest_key


@lambda_error_handler()
def lambda_handler(event, context):
    """
    Persist summary to S3 and metadata to DynamoDB.

    Input:
        - meetingId: str
        - coachName: str
        - employerName: str
        - source: str
        - validatedDataKey: str (S3 key to validated summary data)
        - caseCheckResult: dict (optional)
        - forceReprocess: bool (optional, default False)

    Output:
        - summaryKey: str
        - latestKey: str
        - metadata: dict (for DynamoDB)
    """
    meeting_id = event.get("meetingId")
    coach_name = event.get("coachName")
    employer_name = event.get("employerName", "")
    source = event.get("source")
    validated_data_key = event.get("validatedDataKey")
    force_reprocess = bool(event.get("forceReprocess", False))

    if not meeting_id:
        raise ValidationError("meetingId is required")

    if not validated_data_key:
        raise ValidationError("validatedDataKey is required")

    # Read validated data from S3
    helper.log_json("INFO", "LOADING_VALIDATED_DATA_FROM_S3", meetingId=meeting_id, validatedDataKey=validated_data_key)
    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=validated_data_key)
    validated_summary = json.loads(response['Body'].read().decode('utf-8'))

    if force_reprocess:
        helper.log_json("INFO", "FORCE_REPROCESS_PERSIST", meetingId=meeting_id, coachName=coach_name)

    # Build payload
    payload = _build_summary_payload(
        meeting_id=meeting_id,
        coach_name=coach_name,
        employer_name=employer_name,
        source=source,
        data=validated_summary
    )

    # Save to S3 (force overwrite if force_reprocess=True)
    run_key, latest_key = _save_summary_json(meeting_id, payload, force=force_reprocess)

    # Extract metadata for DynamoDB
    metadata = {
        "sentiment": payload["sentiment"]["label"],
        "actionCount": payload["insights"]["action_count"],
        "themeCount": payload["insights"]["theme_count"],
        "qualityScore": payload["quality_score"]
    }

    return {
        "summaryKey": run_key,
        "latestKey": latest_key,
        "metadata": metadata
    }
