"""
Normalise Roles Lambda - Step Functions workflow step
Normalises speaker names to COACH and CLIENT roles
"""
import re
from utils.error_handler import lambda_error_handler, ValidationError


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
        - transcript: str
        - coachName: str
        - source: str (optional, passed through)

    Output:
        - transcript: str (normalised)
        - source: str (passed through for state optimization)
    """
    transcript = event.get("transcript")
    coach_name = event.get("coachName")
    source = event.get("source")  # Pass through from previous state

    if not transcript:
        raise ValidationError("transcript is required")

    if not coach_name:
        raise ValidationError("coachName is required")

    normalised = normalise_roles(transcript, coach_name)

    return {
        "transcript": normalised,
        "source": source  # Pass through for state optimization
    }
