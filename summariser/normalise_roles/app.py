"""
Normalise Roles Lambda - Step Functions workflow step
Normalises speaker names to COACH and CLIENT roles
"""
import re
import boto3
from utils import helper
from utils.error_handler import lambda_error_handler, ValidationError
from constants import SUMMARY_BUCKET

s3 = boto3.client("s3")


def normalise_roles(transcript: str, coach_name: str) -> str:
    """
    Replace speaker names in transcript with COACH or CLIENT.
    - Matches the coach_name (case-insensitive) and replaces with 'COACH'.
    - All other speakers are replaced with 'CLIENT'.
    - Works with Zoom VTT style: 'Name:' at the start of a line.
    """
    if not transcript or not coach_name:
        return transcript

    coach_pattern = re.compile(rf"^{re.escape(coach_name)}\s*:", re.IGNORECASE)
    speaker_pattern = re.compile(r"^([^:]{2,30}):")  # any 'Name:' up to 30 chars

    out_lines = []
    for line in transcript.splitlines():
        # if line starts with coach name
        if coach_pattern.match(line):
            out_lines.append(coach_pattern.sub("COACH:", line, count=1))
        elif speaker_pattern.match(line):
            out_lines.append(speaker_pattern.sub("CLIENT:", line, count=1))
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


@lambda_error_handler()
def lambda_handler(event, context):
    """
    Normalise speaker roles in transcript.

    Input:
        - transcriptKey: str (S3 key to transcript)
        - meetingId: str
        - coachName: str
        - source: str (optional, passed through)

    Output:
        - transcriptKey: str (S3 key to normalised transcript - same as input)
        - source: str (passed through for state optimization)
    """
    transcript_key = event.get("transcriptKey")
    meeting_id = event.get("meetingId")
    coach_name = event.get("coachName")
    source = event.get("source")  # Pass through from previous state

    if not transcript_key:
        raise ValidationError("transcriptKey is required")

    if not meeting_id:
        raise ValidationError("meetingId is required")

    if not coach_name:
        raise ValidationError("coachName is required")

    # Read transcript from S3
    helper.log_json("INFO", "LOADING_TRANSCRIPT_FROM_S3", meetingId=meeting_id, transcriptKey=transcript_key)
    response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=transcript_key)
    transcript = response['Body'].read().decode('utf-8')

    # Normalise roles
    normalised = normalise_roles(transcript, coach_name)

    # Overwrite the same S3 file with normalised transcript (in-place update)
    s3.put_object(
        Bucket=SUMMARY_BUCKET,
        Key=transcript_key,
        Body=normalised.encode('utf-8'),
        ContentType='text/plain'
    )

    helper.log_json("INFO", "TRANSCRIPT_NORMALISED", meetingId=meeting_id, transcriptKey=transcript_key, size=len(normalised))

    return {
        "transcriptKey": transcript_key,  # Same key, content updated in-place
        "source": source  # Pass through for state optimization
    }
