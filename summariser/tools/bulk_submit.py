#!/usr/bin/env python3
import os, csv, re, time, uuid, json, argparse
import requests

API_BASE      = os.getenv("API_BASE", "https://zrjzpoao4d.execute-api.eu-west-2.amazonaws.com/Prod")
SUMMARISE_URL = f"{API_BASE}/summarise"
STATUS_URL    = f"{API_BASE}/status"

# --- parsing helpers ---
AND_SPLIT = re.compile(r"\s+\band\b\s+", re.IGNORECASE)

def parse_coach_name(session_type: str) -> str:
    """
    Example: 'Octopus Money Starter Session: Leilee Sutton and Nick Forsythe (Santander)'
    -> 'Nick Forsythe'  (second name, strip any trailing '(...)')
    """
    if not session_type:
        return ""
    # take part after ':' if present
    after_colon = session_type.split(":", 1)[-1].strip()
    parts = AND_SPLIT.split(after_colon, maxsplit=1)
    if len(parts) == 2:
        right = parts[1]
    else:
        # fallback: entire tail is coach
        right = after_colon
    # strip trailing parentheses like "(Santander)"
    right = right.split("(")[0].strip()
    # normalise multiple spaces
    right = re.sub(r"\s{2,}", " ", right)
    return right

def normalise_zoom_id(raw: str) -> str:
    return (raw or "").replace(" ", "").strip()

def make_meeting_id(prefix="om") -> str:
    # e.g. om-<uuid4>
    return f"{prefix}-{uuid.uuid4()}"

def submit_job(meeting_id: str, coach_name: str, zoom_meeting_id: str, timeout=15) -> dict:
    payload = {
        "meetingId": meeting_id,
        "coachName": coach_name,
        "zoomMeetingId": zoom_meeting_id,
    }
    r = requests.post(SUMMARISE_URL, json=payload, timeout=timeout)
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text}
    return {"status_code": r.status_code, "response": body, "payload": payload}

def poll_until_done(meeting_id: str, max_secs=120, interval=2) -> dict:
    start = time.time()
    while time.time() - start < max_secs:
        s = requests.get(STATUS_URL, params={"meetingId": meeting_id}, timeout=10)
        try:
            data = s.json()
        except Exception:
            data = {"statusCode": s.status_code, "body": s.text}
        status = (data.get("status") or "").upper()
        if status in ("COMPLETED", "FAILED"):
            return data
        time.sleep(interval)
    return {"status": "TIMEOUT"}

def main():
    ap = argparse.ArgumentParser(description="Bulk submit Octopus Money summaries from CSV")
    ap.add_argument("csv_path", help="Input CSV file")
    ap.add_argument("--host-col", default="Host", help="Column name for Host")
    ap.add_argument("--session-col", default="Session Type", help="Column name for Session Type")
    ap.add_argument("--zoom-col", default="ZOOM ID", help="Column name for Zoom ID")
    ap.add_argument("--out", default="bulk_submissions_out.csv", help="Output CSV mapping file")
    ap.add_argument("--poll", action="store_true", help="Poll each job until COMPLETED/FAILED")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep between submissions (seconds)")
    args = ap.parse_args()

    rows_out = []
    with open(args.csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in [args.session_col, args.zoom_col] if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing column(s) in CSV: {missing}. Found: {reader.fieldnames}")

        for i, row in enumerate(reader, start=1):
            session_type = row.get(args.session_col, "")
            zoom_raw     = row.get(args.zoom_col, "")
            coach_name   = parse_coach_name(session_type)
            zoom_id      = normalise_zoom_id(zoom_raw)
            meeting_id   = make_meeting_id()

            if not coach_name or not zoom_id:
                rows_out.append({
                    "row": i, "meetingId": "", "coachName": coach_name, "zoomMeetingId": zoom_id,
                    "submit_status": "SKIPPED", "reason": "Missing coach or zoom id", "status": ""
                })
                continue

            res = submit_job(meeting_id, coach_name, zoom_id)
            submit_ok = res["status_code"] in (200, 202)
            status = "QUEUED" if submit_ok else f"HTTP_{res['status_code']}"
            poll_status = ""
            if submit_ok and args.poll:
                polled = poll_until_done(meeting_id)
                poll_status = polled.get("status", "")
                status = poll_status or status

            rows_out.append({
                "row": i,
                "meetingId": meeting_id,
                "coachName": coach_name,
                "zoomMeetingId": zoom_id,
                "submit_status": "OK" if submit_ok else "ERROR",
                "status": status,
            })

            print(f"[{i}] {meeting_id} :: coach='{coach_name}' zoom='{zoom_id}' -> {status}")
            time.sleep(args.sleep)

    # write a simple mapping file so you can look up meetingIds in the dashboard
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row","meetingId","coachName","zoomMeetingId","submit_status","status"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} rows to {args.out}")
    print("Tip: open the Dashboard tab or call GET /summaries to see items as they complete.")

if __name__ == "__main__":
    main()
