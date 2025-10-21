"""
Fetch Transcript Lambda - Step Functions workflow step
Fetches transcript from either direct input, S3, or Zoom API
"""
import json
import os
from typing import Tuple
import time
import base64
import requests
import re
from datetime import datetime, timezone

import boto3
from utils.error_handler import lambda_error_handler, ValidationError
from constants import *

s3 = boto3.client("s3")
ssm = boto3.client("ssm")

VTT_TS = re.compile(r'^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}$')


def _ssm_get(name, decrypt=False):
    return ssm.get_parameter(
        Name=f"{ZOOM_PARAM_PREFIX}/{name}", WithDecryption=decrypt
    )["Parameter"]["Value"]


def _ssm_put(name, value, secure=False):
    kwargs = dict(
        Name=f"{ZOOM_PARAM_PREFIX}/{name}", Value=value, Overwrite=True
    )
    kwargs["Type"] = "SecureString" if secure else "String"
    ssm.put_parameter(**kwargs)


def _get_zoom_token_from_ssm():
    """Get or refresh Zoom access token from SSM"""
    try:
        tok = _ssm_get("access_token", True)
        exp = int(_ssm_get("access_token_expires_at"))
        if time.time() < exp - 60:
            return tok
    except Exception:
        pass

    acct = _ssm_get("account_id")
    cid = _ssm_get("client_id", True)
    sec = _ssm_get("client_secret", True)
    basic = base64.b64encode(f"{cid}:{sec}".encode()).decode()
    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={acct}"

    r = requests.post(
        url,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    tok = data["access_token"]
    exp = int(time.time()) + int(data.get("expires_in", 3600))
    _ssm_put("access_token", tok, True)
    _ssm_put("access_token_expires_at", str(exp), False)
    return tok


def vtt_to_text(vtt: str) -> str:
    """Strip WEBVTT headers, cue numbers, timestamps and collapse blanks"""
    out = []
    for line in vtt.splitlines():
        s = line.strip()
        if not s:
            out.append("")
            continue
        if s.upper().startswith("WEBVTT"):
            continue
        if VTT_TS.match(s):
            continue
        if s.isdigit():
            continue
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def _s3_put_text(bucket: str, key: str, text: str, content_type="text/plain"):
    """Save text to S3"""
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)


def _retry_http(fn, tries=4, base=0.5):
    """Retry HTTP requests with exponential backoff"""
    for i in range(tries):
        try:
            return fn()
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", 0)
            if code in (429, 500, 502, 503, 504):
                time.sleep(base * (2 ** i))
                continue
            raise


def fetch_transcript_by_zoom_meeting_id(zoom_meeting_id: str, meeting_id: str) -> Tuple[str, str]:
    """
    Fetch transcript from Zoom API and optionally save to S3.
    Returns (plain_text_transcript, raw_vtt).
    """
    token = _get_zoom_token_from_ssm()

    def _list_files():
        url = f"https://api.zoom.us/v2/meetings/{zoom_meeting_id}/recordings"
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=15)
        if r.status_code == 404:
            raise RuntimeError("Zoom recordings not found yet (still processing).")
        r.raise_for_status()
        return r.json().get("recording_files", []) or []

    files = _retry_http(_list_files)

    vtt_url = None
    for f in files:
        if f.get("file_type") in ("TRANSCRIPT", "CC"):
            vtt_url = f.get("download_url")
            break
    if not vtt_url:
        raise RuntimeError("No transcript file (TRANSCRIPT/CC) available for this meeting.")

    def _download_vtt():
        r = requests.get(vtt_url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
        if r.status_code == 401:
            r = requests.get(f"{vtt_url}?access_token={token}", timeout=20)
        r.raise_for_status()
        return r.text

    raw_vtt = _retry_http(_download_vtt)
    plain = vtt_to_text(raw_vtt)

    # Optional S3 persistence
    if SAVE_TRANSCRIPTS:
        now = datetime.now(timezone.utc)
        if ATHENA_PARTITIONED:
            base = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={now.year}/month={now.month:02d}/meeting_id={meeting_id}"
        else:
            base = f"{S3_PREFIX}/{now:%Y}/{now:%m}/{meeting_id}"

        _s3_put_text(SUMMARY_BUCKET, f"{base}/zoom_raw.vtt", raw_vtt, "text/vtt")
        _s3_put_text(SUMMARY_BUCKET, f"{base}/transcript.txt", plain, "text/plain")

    return plain, raw_vtt


def fetch_from_s3(meeting_id: str) -> str:
    """Fetch existing transcript from S3 for reprocessing"""
    possible_paths = [
        f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year=2025/month=09/meeting_id={meeting_id}/transcript.txt",
        f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year=2025/month=10/meeting_id={meeting_id}/transcript.txt",
        f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year=2025/month=08/meeting_id={meeting_id}/transcript.txt",
    ]

    for path in possible_paths:
        try:
            response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=path)
            return response['Body'].read().decode('utf-8')
        except Exception:
            continue

    raise ValueError(f"Could not find existing transcript for meeting {meeting_id}")


@lambda_error_handler()
def lambda_handler(event, context):
    """
    Fetch transcript from direct input, Zoom API, or S3.

    Input:
        - meetingId: str
        - transcript: str (optional)
        - zoomMeetingId: str (optional)
        - coachName: str
        - employerName: str

    Output:
        - transcript: str
        - source: str (direct|zoom_api|s3_existing)
        - rawVtt: str (optional, only for Zoom)
    """
    meeting_id = event.get("meetingId")
    transcript = event.get("transcript")
    zoom_meeting_id = event.get("zoomMeetingId")

    if not meeting_id:
        raise ValidationError("meetingId is required")

    # Direct transcript provided
    if transcript:
        # Save to S3 if enabled
        if SAVE_TRANSCRIPTS:
            now = datetime.now(timezone.utc)
            if ATHENA_PARTITIONED:
                base = f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/year={now.year}/month={now.month:02d}/meeting_id={meeting_id}"
            else:
                base = f"{S3_PREFIX}/{now:%Y}/{now:%m}/{meeting_id}"
            _s3_put_text(SUMMARY_BUCKET, f"{base}/transcript.txt", transcript, "text/plain")

        return {
            "transcript": transcript,
            "source": "direct",
            "rawVtt": None
        }

    # Fetch from Zoom
    elif zoom_meeting_id:
        plain, raw_vtt = fetch_transcript_by_zoom_meeting_id(zoom_meeting_id, meeting_id)
        return {
            "transcript": plain,
            "source": "zoom_api",
            "rawVtt": raw_vtt
        }

    # Fetch from S3 (reprocessing)
    else:
        transcript = fetch_from_s3(meeting_id)
        return {
            "transcript": transcript,
            "source": "s3_existing",
            "rawVtt": None
        }
