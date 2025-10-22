"""
PII Detect & Redact Lambda - Step Functions workflow step
Uses AWS Comprehend to detect and redact PII entities
"""
import boto3
import os
from datetime import datetime, timezone
from utils import helper
from utils.error_handler import lambda_error_handler, ValidationError
from constants import SUMMARY_BUCKET, S3_PREFIX, SCHEMA_VERSION, ATHENA_PARTITIONED

comprehend = boto3.client("comprehend")
s3 = boto3.client("s3")


def pii_entities_chunked(text: str, chunk=4500, overlap=200, min_score=0.7, mask_types=None) -> list:
    """
    Run Comprehend PII detection over long text by sliding window.
    Returns deduped entities with offsets rebased to original text.
    """
    entities = []
    i, n = 0, len(text)

    while i < n:
        part = text[i:i+chunk]
        resp = comprehend.detect_pii_entities(Text=part, LanguageCode="en")
        for e in resp.get("Entities", []):
            if e["Score"] < min_score:
                continue
            if mask_types and e["Type"] not in mask_types:
                continue
            entity = dict(e)
            entity["BeginOffset"] += i
            entity["EndOffset"] += i
            entities.append(entity)
        i += max(1, chunk - overlap)

    # dedupe by offset+type
    seen, deduped = set(), []
    for e in entities:
        key = (e["BeginOffset"], e["EndOffset"], e["Type"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    return deduped


def redact_text(text: str, entities: list) -> str:
    """Redact text by replacing PII entities with [REDACTED]"""
    redacted = list(text)
    # Replace from the end to avoid offset shifts
    for entity in sorted(entities, key=lambda x: x["BeginOffset"], reverse=True):
        redacted[entity["BeginOffset"]:entity["EndOffset"]] = "[REDACTED]"
    return "".join(redacted)


@lambda_error_handler()
def lambda_handler(event, context):
    """
    Detect and redact PII entities in transcript.

    Input:
        - transcriptKey: str (S3 key to transcript)
        - meetingId: str
        - source: str (optional, passed through)

    Output:
        - meetingId: str (passed through for state optimization)
        - source: str (passed through for state optimization)
        - redactedTranscriptKey: str (S3 key to redacted transcript)
        - piiEntityCount: int
    """
    transcript_key = event.get("transcriptKey")
    meeting_id = event.get("meetingId")
    source = event.get("source")  # Pass through from previous state

    if not transcript_key:
        raise ValidationError("transcriptKey is required")

    if not meeting_id:
        raise ValidationError("meetingId is required")

    # Read transcript from S3
    helper.log_json("INFO", "LOADING_TRANSCRIPT_FROM_S3", meetingId=meeting_id, transcriptKey=transcript_key)
    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=transcript_key)
    transcript = response['Body'].read().decode('utf-8')

    # Detect PII entities
    pii_entities = pii_entities_chunked(transcript)

    # Redact transcript
    redacted_transcript = redact_text(transcript, pii_entities)

    # Save redacted transcript to S3 to avoid Step Functions state size limit
    now = datetime.now(timezone.utc)
    if ATHENA_PARTITIONED:
        s3_key = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={now.year}/month={now.month:02d}/meeting_id={meeting_id}/redacted_transcript.txt"
    else:
        s3_key = f"{S3_PREFIX}/{now:%Y}/{now:%m}/{meeting_id}/redacted_transcript.txt"

    s3.put_object(
        Bucket=SUMMARY_BUCKET,
        Key=s3_key,
        Body=redacted_transcript.encode("utf-8"),
        ContentType="text/plain"
    )

    return {
        "meetingId": meeting_id,  # Pass through for state optimization
        "source": source,  # Pass through for state optimization
        "redactedTranscriptKey": s3_key,
        "piiEntityCount": len(pii_entities)
        # Removed piiEntities array to reduce state size (not used downstream)
        # If needed for analytics, log to CloudWatch or save separately to S3
    }
