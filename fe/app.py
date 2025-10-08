# fe/app.py
import os
import time
import json
import math
import html
import re
import requests
import gradio as gr
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
load_dotenv()

print("ENV CHECK:", {
    "API_BASE": os.getenv("API_BASE"),
    "ATHENA_REGION": os.getenv("ATHENA_REGION"),
    "ATHENA_WORKGROUP": os.getenv("ATHENA_WORKGROUP"),
    "ATHENA_S3_STAGING": os.getenv("ATHENA_S3_STAGING"),
    "ATHENA_DATABASE": os.getenv("ATHENA_DATABASE"),
    "S3_PREFIX": os.getenv("S3_PREFIX"),
})

# =========================================================
# Config (API)
# =========================================================
API_BASE            = os.getenv("API_BASE", "https://zrjzpoao4d.execute-api.eu-west-2.amazonaws.com/Prod")
SUMMARISE_URL       = f"{API_BASE}/summarise"
STATUS_URL          = f"{API_BASE}/status"
SUMMARIES_URL       = f"{API_BASE}/summaries"
CASE_URL            = f"{API_BASE}/case"
A2I_PORTAL_URL      = os.getenv("A2I_PORTAL_URL", "https://29cw9ax36n.labeling.eu-west-2.sagemaker.aws/")
POLL_INTERVAL_SECS  = int(os.getenv("POLL_INTERVAL_SECS", "2"))
POLL_TIMEOUT_SECS   = int(os.getenv("POLL_TIMEOUT_SECS", "180"))
DEFAULT_PAGE_SIZE   = int(os.getenv("DEFAULT_PAGE_SIZE", "20"))
S3_PREFIX           = os.getenv("S3_PREFIX", "summaries")  # e.g. 'summaries/v=1.1'

# =========================================================
# Config (Athena). Leave empty to hide the Insights tab.
# =========================================================
ATHENA_REGION      = os.getenv("ATHENA_REGION", "eu-west-2")
ATHENA_WORKGROUP   = os.getenv("ATHENA_WORKGROUP", "primary")
ATHENA_S3_STAGING  = os.getenv("ATHENA_S3_STAGING")  # e.g. s3://bucket/athena-results/
ATHENA_DATABASE    = os.getenv("ATHENA_DATABASE", "call_summaries")

# Lazy import (so the rest of the app works even if PyAthena isn't configured)
PYATHENA_OK = True
try:
    from pyathena import connect as _athena_connect
    from pyathena.pandas.cursor import PandasCursor
except Exception:
    PYATHENA_OK = False

# =========================================================
# Submit flow (single button, coach optional)
# =========================================================
def _fetch_status(meeting_id: str):
    try:
        s = requests.get(STATUS_URL, params={"meetingId": meeting_id}, timeout=10)
        if s.status_code != 200:
            return None, f"status HTTP {s.status_code}"
        return s.json(), None
    except Exception as e:
        return None, str(e)

def _download_summary(download_url: str):
    r = requests.get(download_url, timeout=20)
    if r.status_code != 200:
        return None, f"download HTTP {r.status_code}"
    try:
        return json.dumps(r.json(), indent=2), None
    except Exception:
        return r.text, None

def submit_and_get_json(meeting_id: str, coach_name: str, employer_name: str, transcript: str, zoom_meeting_id: str, enable_case_check: bool = False) -> str:
    meeting_id = (meeting_id or "").strip()
    coach_name = (coach_name or "").strip()
    employer_name = (employer_name or "").strip()

    if not meeting_id:
        return json.dumps({"error": "Please provide Meeting ID (OM)."}, indent=2)

    # 1) If exists and completed, just fetch it
    status, err = _fetch_status(meeting_id)
    if status and (status.get("status") or "").upper() == "COMPLETED":
        url = status.get("downloadUrl")
        if not url:
            return json.dumps({"error": "Completed but no downloadUrl found."}, indent=2)
        body, derr = _download_summary(url)
        return body if derr is None else json.dumps({"error": derr}, indent=2)

    # 2) Otherwise, (re)submit a job — coach name optional, but useful.
    if not transcript and not zoom_meeting_id:
        need = []
        if not transcript: need.append("Transcript")
        if not zoom_meeting_id: need.append("Zoom Meeting ID")
        return json.dumps({"error": "This meeting hasn't completed yet.",
                           "need": need,
                           "hint": "Provide a Transcript or a Zoom Meeting ID to process."}, indent=2)

    payload = {"meetingId": meeting_id, "coachName": coach_name, "employerName": employer_name}

    # Add case checking option
    if enable_case_check:
        payload["enableCaseCheck"] = True

    print(f"Submitting payload with case checking {'enabled' if enable_case_check else 'disabled'}: {json.dumps(payload)}")
    if transcript and transcript.strip():
        payload["transcript"] = transcript.strip()
    else:
        payload["zoomMeetingId"] = (zoom_meeting_id or "").replace(" ", "")

    try:
        r = requests.post(SUMMARISE_URL, json=payload, timeout=15)
        if r.status_code not in (200, 202):
            return json.dumps({"error": "Submit failed", "status": r.status_code, "body": r.text}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Submit exception: {e}"}, indent=2)

    # 3) Poll status
    start = time.time()
    while time.time() - start < POLL_TIMEOUT_SECS:
        st, _ = _fetch_status(meeting_id)
        if st and (st.get("status") or "").upper() == "COMPLETED":
            url = st.get("downloadUrl")
            if not url:
                return json.dumps({"error": "No downloadUrl in status response."}, indent=2)
            body, derr = _download_summary(url)
            return body if derr is None else json.dumps({"error": derr}, indent=2)
        if st and (st.get("status") or "").upper() == "FAILED":
            return json.dumps({"error": "Processing failed", "details": st.get("error")}, indent=2)
        time.sleep(POLL_INTERVAL_SECS)

    return json.dumps({"error": f"Timed out after {POLL_TIMEOUT_SECS}s"}, indent=2)

# =========================================================
# Dashboard helpers (API-backed) — version-aware
# =========================================================
def fetch_summaries_json(version: str | None = None) -> dict:
    """Fetch summaries data directly from Athena since the API is broken"""
    try:
        if not PYATHENA_OK or not ATHENA_S3_STAGING:
            # Fallback to API if Athena not available
            params = {"version": (version or S3_PREFIX)}
            r = requests.get(SUMMARIES_URL, params=params, timeout=20)
            if r.status_code != 200:
                return {"error": "Failed to fetch summaries", "status": r.status_code, "body": r.text}
            return r.json()

        # Only support schema version 1.2
        schema_version = "1.2"

        # Query Athena directly for the data (updated for new Parquet structure)
        sql = f"""
        SELECT
            meeting_id as meetingId,
            coach_name as coachName,
            employer_name as employerName,
            sentiment_label as sentiment,
            summary,
            themes,
            actions,
            schema_version as version
        FROM {ATHENA_DATABASE}.summaries
        WHERE schema_version = '{schema_version}'
        LIMIT 1000
        """

        df = run_df(sql)

        # Convert DataFrame to the expected format
        items = []
        for _, row in df.iterrows():
            # Handle NaN values properly
            item = {
                "meetingId": str(row.get("meetingId", "")),
                "coachName": str(row.get("coachName", "")) if pd.notna(row.get("coachName")) else "",
                "employerName": str(row.get("employerName", "")) if pd.notna(row.get("employerName")) else "",
                "sentiment": str(row.get("sentiment", "")) if pd.notna(row.get("sentiment")) else "",
                "summary": str(row.get("summary", "")) if pd.notna(row.get("summary")) else "",
                "themes": row.get("themes", []) if pd.notna(row.get("themes")) else [],
                "actions": row.get("actions", []) if pd.notna(row.get("actions")) else [],
                "version": str(row.get("version", "")),
                "updatedAt": "2025-09-23T00:00:00Z"  # placeholder
            }
            items.append(item)

        return {"items": items, "total": len(items)}

    except Exception as e:
        return {"error": str(e)}

def fetch_summaries_from_athena(version: str) -> dict:
    """Fetch summaries data from Athena for the specified version"""
    try:
        if not PYATHENA_OK or not ATHENA_S3_STAGING:
            return {"error": "Athena not configured. Set ATHENA_* env vars."}

        # Query to fetch DISTINCT summaries data from Athena with actual pass rates from enhanced table
        sql = f"""
        SELECT DISTINCT
            meeting_id as meetingId,
            coach_name as coachName,
            employer_name as employerName,
            sentiment_label as sentiment,
            schema_version as version,
            '2025-09-24T00:00:00Z' as updatedAt,
            case_pass_rate as casePassRate,
            case_failed_count as caseFailedCount,
            CAST(cardinality(actions) AS INTEGER) as actionCount,
            sentiment_confidence as qualityScore,
            CASE
                WHEN case_pass_rate < 0.3 THEN 'HIGH'
                WHEN case_pass_rate < 0.6 THEN 'MEDIUM'
                ELSE 'LOW'
            END as riskLevel,
            CAST(NULL AS DOUBLE) as vulnerabilityScore,
            CASE
                WHEN case_pass_rate < 0.3 THEN 'HIGH'
                WHEN case_pass_rate < 0.6 THEN 'MEDIUM'
                ELSE 'LOW'
            END as severityLevel,
            CAST(CASE WHEN case_pass_rate IS NOT NULL THEN true ELSE false END AS BOOLEAN) as caseCheckEnabled,
            CAST(NULL AS VARCHAR) as s3Key
        FROM {ATHENA_DATABASE}.summaries
        WHERE schema_version = '1.2'
        LIMIT 1000
        """

        df = run_df(sql)

        # Get employer data from DynamoDB to supplement Athena data
        import boto3
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('summary-job-status')

        employer_data = {}
        try:
            response = table.scan()
            items_ddb = response['Items']

            while 'LastEvaluatedKey' in response:
                response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                items_ddb.extend(response['Items'])

            # Create mapping of meetingId -> employerName
            for item in items_ddb:
                meeting_id = item.get('meetingId')
                employer_name = item.get('employerName')
                if meeting_id and employer_name:
                    employer_data[meeting_id] = employer_name

        except Exception as e:
            pass

        # Convert DataFrame to the format expected by the dashboard
        items = []
        for _, row in df.iterrows():
            meeting_id = row.get("meetingId")
            # Get employer name from DynamoDB if available
            employer_name = employer_data.get(meeting_id, row.get("employerName"))

            item = {
                "meetingId": meeting_id,
                "coachName": row.get("coachName"),
                "employerName": employer_name,
                "sentiment": row.get("sentiment"),
                "updatedAt": row.get("updatedAt"),
                "casePassRate": row.get("casePassRate"),
                "caseFailedCount": row.get("caseFailedCount"),
                "actionCount": row.get("actionCount"),
                "qualityScore": row.get("qualityScore"),
                "riskLevel": row.get("riskLevel"),
                "vulnerabilityScore": row.get("vulnerabilityScore"),
                "severityLevel": row.get("severityLevel"),
                "caseCheckEnabled": row.get("caseCheckEnabled"),
                "version": row.get("version"),
                "s3Key": row.get("s3_key")
            }
            items.append(item)

        return {"items": items}

    except Exception as e:
        return {"error": f"Athena query failed: {str(e)}"}

def chip(text, color="#e2e8f0", fg="#111827"):
    return f'<span style="display:inline-block;padding:2px 8px;border-radius:9999px;background:{color};color:{fg};font-size:12px;">{html.escape(str(text))}</span>'

def render_summary_table(items: list) -> str:
    if not items:
        return "<p>No completed items yet for this version.</p>"

    header = """
    <thead>
      <tr>
        <th style="text-align:left;">Updated</th>
        <th style="text-align:left;">Meeting ID</th>
        <th style="text-align:left;">Employer</th>
        <th style="text-align:left;">Sentiment</th>
        <th style="text-align:right;">Actions</th>
        <th style="text-align:right;">Case Pass %</th>
        <th style="text-align:right;">Case Fails</th>
        <th style="text-align:left;">Review</th>
        <th style="text-align:left;">Version</th>
        <th style="text-align:left;">Summary JSON</th>
        <th style="text-align:left;">Case JSON</th>
      </tr>
    </thead>"""
    rows = []
    for it in items:
        updated   = html.escape(str(it.get("updatedAt","")))
        mid       = html.escape(str(it.get("meetingId","")))
        sent      = html.escape(str(it.get("sentiment","")))
        employer  = html.escape(str(it.get("employerName") or "No employer"))
        version   = html.escape(str(it.get("prefixVersion") or ""))
        acnt      = it.get("actionCount")
        passrate  = it.get("casePassRate")
        fails     = it.get("caseFailedCount")
        status    = (it.get("status") or "").upper()
        loop      = it.get("a2iCaseLoop") or ""
        s_url     = it.get("summaryUrl")
        c_url     = it.get("caseUrl")

        acnt_str  = "" if acnt is None else f"{acnt}"
        pr_str    = "" if passrate is None else f"{round(float(passrate)*100,1)}%"
        fails_str = "" if fails is None else f"{fails}"

        # Review cell
        if status == "IN_REVIEW":
            badge = chip("In review", "#fef3c7", "#92400e")
            portal = f' — <a href="{html.escape(A2I_PORTAL_URL)}" target="_blank">Open portal</a>' if A2I_PORTAL_URL else ""
            loop_html = f'<div style="font-family:ui-monospace,monospace;font-size:12px;color:#6b7280;">{html.escape(str(loop))}</div>' if loop else ""
            review_cell = f'{badge}{portal}{loop_html}'
        elif status == "FAILED":
            review_cell = chip("Failed", "#fee2e2", "#991b1b")
        else:
            review_cell = chip("Completed", "#dcfce7", "#065f46")

        s_link = f'<a href="{html.escape(s_url)}" target="_blank">Open</a>' if s_url else ""
        c_link = f'<a href="{html.escape(c_url)}" target="_blank">Open</a>' if c_url else ""

        rows.append(f"""
          <tr>
            <td>{updated}</td>
            <td>{mid}</td>
            <td>{employer}</td>
            <td>{sent}</td>
            <td style="text-align:right;">{acnt_str}</td>
            <td style="text-align:right;">{pr_str}</td>
            <td style="text-align:right;">{fails_str}</td>
            <td>{review_cell}</td>
            <td>{version}</td>
            <td>{s_link}</td>
            <td>{c_link}</td>
          </tr>""")

    return f"""
    <table style="border-collapse:collapse; width:100%; font-family: ui-sans-serif, system-ui; font-size: 14px;">
      {header}
      <tbody>{''.join(rows)}</tbody>
    </table>"""

def _paginate(items, page, page_size):
    total = len(items)
    page_size = max(1, int(page_size or DEFAULT_PAGE_SIZE))
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(int(page or 1), total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    return items[start:end], page, total_pages, total, page_size

def render_existing(items, page, page_size):
    page_items, page, total_pages, total, page_size = _paginate(items, page, page_size)
    table_html = render_summary_table(page_items)
    page_label = f"Page {page} of {total_pages} ({total} items)"
    return page, table_html, json.dumps(page_items, indent=2), page_label

def _filter_complete_records(items: list, version: str = None) -> list:
    """Filter items to only include records with complete compliance data"""
    complete_items = []

    for item in items:
        # For v=1.2, expect complete compliance data
        if version == "summaries/v=1.2":
            # Require essential compliance fields for v=1.2
            has_compliance = (
                item.get("casePassRate") is not None or
                item.get("caseFailedCount") is not None or
                item.get("actionCount") is not None
            )
            # Also require basic meeting data
            has_basic_data = (
                item.get("meetingId") and
                item.get("sentiment") and
                item.get("updatedAt")
            )

            if has_compliance and has_basic_data:
                complete_items.append(item)
        else:
            # For other versions, be more lenient
            if item.get("meetingId"):
                complete_items.append(item)

    return complete_items

def _kpis_from_items(items: list):
    if not items:
        return 0, 0.0, 0.0
    actions = [float(it.get("actionCount")) for it in items if isinstance(it.get("actionCount"), (int, float))]
    passrates = [float(it.get("casePassRate")) for it in items if isinstance(it.get("casePassRate"), (int, float))]
    return len(items), (sum(actions)/len(actions) if actions else 0.0), (sum(passrates)/len(passrates) if passrates else 0.0)

def _sentiment_df(items: list) -> pd.DataFrame:
    if not items:
        return pd.DataFrame({"sentiment": [], "count": []})
    counts = {}
    for it in items:
        s = it.get("sentiment") or "Unknown"
        counts[s] = counts.get(s, 0) + 1
    return pd.DataFrame({"sentiment": list(counts.keys()), "count": list(counts.values())})

def _extract_versions(items: list) -> list[str]:
    vs = sorted({it.get("prefixVersion") for it in items if it.get("prefixVersion")})
    return vs

def _simple_kpis():
    """Get just the essential KPIs without complex tables"""
    data = fetch_summaries_from_athena(S3_PREFIX)
    if "error" in data:
        return 0, "0.00", 0, 0

    items = data.get("items", [])
    total, avg_actions, _ = _kpis_from_items(items)

    # Count escalation candidates and negative sentiment
    escalations = sum(1 for item in items if _is_escalation_candidate(item))
    negative_sentiment = sum(1 for item in items if (item.get("sentiment") or "").lower() == "negative")

    return total, f"{avg_actions:.2f}", escalations, negative_sentiment

def _is_escalation_candidate(item):
    """Check if an item is an escalation candidate based on various factors"""
    meeting_id = item.get("meetingId", "UNKNOWN")
    escalated = False
    reason = None

    # Check if explicitly marked as escalation candidate
    if item.get("is_escalation_candidate"):
        escalated = True
        reason = f"Explicitly marked: {item.get('is_escalation_candidate')}"

    # Check negative sentiment
    if not escalated:
        sentiment = (item.get("sentiment") or "").lower()
        if sentiment == "negative":
            escalated = True
            reason = f"Negative sentiment: '{sentiment}'"

    # Check low case pass rate (if available)
    if not escalated:
        pass_rate = item.get("casePassRate")
        if pass_rate is not None and float(pass_rate) < 0.5:
            escalated = True
            reason = f"Low pass rate: {pass_rate} < 0.5"

    # Check high case failure count
    if not escalated:
        fail_count = item.get("caseFailedCount")
        if fail_count is not None and int(fail_count) >= 3:
            escalated = True
            reason = f"High failure count: {fail_count} >= 3"

    # Debug logging for escalated meetings
    if escalated:
        print(f"ESCALATION: Meeting {meeting_id} flagged - {reason}")
        print(f"  Full data: sentiment='{item.get('sentiment')}', casePassRate={item.get('casePassRate')}, caseFailedCount={item.get('caseFailedCount')}, is_escalation_candidate={item.get('is_escalation_candidate')}")

    return escalated

def load_themes_from_api():
    """Extract themes from Athena data for the specified version"""
    try:
        # Use the _ver helper to convert version formats like "summaries/v=1.2" to "1.2"
        schema_version = "1.2"

        # Query Athena directly to avoid pandas/PyAthena compatibility issues
        import boto3
        import time

        athena = boto3.client('athena')
        S3_RESULTS = 's3://call-summariser-summarybucket-3wtnjhb9vvq0/athena-results/'

        sql = f"""
        SELECT themes, schema_version
        FROM call_summaries.summaries
        WHERE schema_version = '{schema_version}'
        AND themes IS NOT NULL
        AND cardinality(themes) > 0
        LIMIT 100
        """

        # Execute Athena query directly
        response = athena.start_query_execution(
            QueryString=sql,
            WorkGroup='primary',
            ResultConfiguration={'OutputLocation': S3_RESULTS}
        )

        query_id = response['QueryExecutionId']

        # Wait for completion
        while True:
            result = athena.get_query_execution(QueryExecutionId=query_id)
            status = result['QueryExecution']['Status']['State']
            if status == 'SUCCEEDED':
                break
            elif status in ['FAILED', 'CANCELLED']:
                raise Exception(f'Query failed')
            time.sleep(1)

        # Get results
        results = athena.get_query_results(QueryExecutionId=query_id)
        rows = results['ResultSet']['Rows']

        # Convert to DataFrame manually
        if len(rows) <= 1:
            df = pd.DataFrame()
        else:
            columns = [col['VarCharValue'] for col in rows[0]['Data']]
            data = []
            for row in rows[1:]:
                row_data = []
                for cell in row['Data']:
                    row_data.append(cell.get('VarCharValue', ''))
                data.append(row_data)
            df = pd.DataFrame(data, columns=columns)
        if df.empty:
            return px.bar(title=f"No theme data found for version {schema_version}")

        theme_counts = {}

        # Process themes from DataFrame
        for _, row in df.iterrows():
            themes_raw = row.get("themes", "[]")
            if pd.isna(themes_raw) or themes_raw == "[]":
                continue

            try:
                import json
                themes = json.loads(themes_raw) if isinstance(themes_raw, str) else themes_raw

                if isinstance(themes, list):
                    for theme in themes:
                        if isinstance(theme, dict):
                            theme_name = theme.get("label") or theme.get("id", "Unknown")
                            theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1
                        elif isinstance(theme, str):
                            theme_counts[theme] = theme_counts.get(theme, 0) + 1

            except Exception as e:
                continue

        if not theme_counts:
            return px.bar(title=f"No themes extracted for version {schema_version}")

        print("Theme count 123 ", theme_counts)
        # Create chart data
        theme_data = pd.DataFrame([
            {"theme": theme, "count": count}
            for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ])

        print("DataFrame shape:", theme_data.shape)
        print("DataFrame head:", theme_data.head())
        print("DataFrame content:", theme_data.to_dict('records'))

        fig = px.bar(
            theme_data,
            x="theme",
            y="count",
            title=f"Top Discussion Themes (v{schema_version})",
            labels={"count": "Mentions", "theme": "Theme"}
        )
        fig.update_layout(xaxis_tickangle=45)

        print("Chart created, data points:", len(fig.data[0].x) if fig.data else 0)
        print("Returning chart successfully")
        return fig

    except Exception as e:
        print(f"ERROR in load_themes_from_api: {e}")
        import traceback
        traceback.print_exc()
        return px.bar(title=f"Error loading themes: {str(e)[:100]}")

def load_quality_from_api():
    """Extract quality scores from API data when Athena is unavailable"""
    try:
        data = fetch_summaries_from_athena(S3_PREFIX)
        if "error" in data:
            return px.bar(title="Error loading quality data")

        items = data.get("items", [])
        coach_data = {}

        # Group by coach and calculate averages
        for item in items:
            coach = item.get("coach", "Unknown")
            if coach not in coach_data:
                coach_data[coach] = {"scores": [], "actions": [], "meetings": 0}

            coach_data[coach]["meetings"] += 1

            # Use action count as a proxy for quality (higher = better)
            action_count = item.get("actionCount", 0)
            if action_count is not None:
                coach_data[coach]["actions"].append(action_count)

        if not coach_data:
            return px.bar(title="No coach data available")

        # Calculate averages
        coaches = []
        avg_actions = []
        meeting_counts = []

        for coach, data in coach_data.items():
            if data["actions"]:  # Only include coaches with action data
                coaches.append(coach)
                avg_actions.append(sum(data["actions"]) / len(data["actions"]))
                meeting_counts.append(data["meetings"])

        if not coaches:
            return px.bar(title="No quality score data available")

        df = pd.DataFrame({
            "coach": coaches,
            "avg_actions": avg_actions,
            "meeting_count": meeting_counts
        })

        return px.bar(df, x="coach", y="avg_actions", title="Coach Performance (Avg Actions per Meeting)",
                     hover_data=["meeting_count"])

    except Exception as e:
        return px.bar(title=f"Error: {str(e)[:50]}")

def _create_empty_bar_chart(title: str):
    """Create an empty bar chart that Plotly can render"""
    empty_df = pd.DataFrame({"category": ["No Data"], "count": [0]})
    return px.bar(empty_df, x="category", y="count", title=title)

def _create_empty_pie_chart(title: str):
    """Create an empty pie chart that Plotly can render"""
    empty_df = pd.DataFrame({"category": ["No Data"], "count": [1]})
    return px.pie(empty_df, values="count", names="category", title=title)

def load_version_specific_analytics():
    """Load all analytics charts for a specific version"""
    version = "summaries/v=1.2"  # Only v1.2 supported
    try:
        data = fetch_summaries_from_athena(version)
        if "error" in data:
            empty_fig = _create_empty_bar_chart(f"Error loading data for version {version}")
            empty_pie = _create_empty_pie_chart(f"Error loading data for version {version}")
            return empty_fig, empty_pie, empty_fig, empty_fig

        raw_items = data.get("items", [])
        # Filter to only include complete records
        items = _filter_complete_records(raw_items, version)

        if not items:
            empty_fig = _create_empty_bar_chart(f"No complete data available for version {version}")
            empty_pie = _create_empty_pie_chart(f"No complete data available for version {version}")
            return empty_fig, empty_pie, empty_fig, empty_fig

        # Coach Performance Chart
        try:
            coach_performance_fig = _create_coach_performance_chart(items, version)
        except Exception:
            coach_performance_fig = _create_empty_bar_chart(f"Coach Performance - Version {version} (No Data)")

        # Sentiment Analysis Chart
        try:
            sentiment_fig = _create_sentiment_chart(items, version)
        except Exception:
            sentiment_fig = _create_empty_pie_chart(f"Sentiment Distribution - Version {version} (No Data)")

        # Theme Trends Chart
        try:
            themes_fig = load_themes_from_api()
        except Exception:
            themes_fig = _create_empty_bar_chart(f"Top Discussion Themes - Version {version} (No Data)")

        # Risk & Vulnerability Chart
        try:
            risk_fig = _create_risk_chart(items, version)
        except Exception:
            risk_fig = _create_empty_bar_chart(f"Risk & Vulnerability Assessment - Version {version} (No Data)")

        return coach_performance_fig, sentiment_fig, themes_fig, risk_fig

    except Exception as e:
        error_fig = _create_empty_bar_chart(f"Error: {str(e)[:50]}")
        error_pie = _create_empty_pie_chart(f"Error: {str(e)[:50]}")
        return error_fig, error_pie, error_fig, error_fig

def _create_coach_performance_chart(items, version):
    """Create coach performance chart"""
    coach_data = {}
    for item in items:
        # Try multiple possible coach name fields
        coach = (item.get("coach") or
                item.get("coachName") or
                item.get("coach_name") or
                item.get("employerName", "").split()[0] if item.get("employerName") else None or
                f"Coach-{len(coach_data) + 1}")  # Generate coach names if missing

        if coach == "Unknown" or not coach.strip():
            coach = f"Coach-{len(coach_data) + 1}"

        if coach not in coach_data:
            coach_data[coach] = {"actions": [], "pass_rates": [], "meetings": 0, "sentiments": []}

        coach_data[coach]["meetings"] += 1

        action_count = item.get("actionCount", 0)
        if action_count is not None and action_count > 0:
            coach_data[coach]["actions"].append(action_count)

        pass_rate = item.get("casePassRate")
        if pass_rate is not None:
            coach_data[coach]["pass_rates"].append(float(pass_rate) * 100)

        sentiment = item.get("sentiment", "")
        if sentiment:
            coach_data[coach]["sentiments"].append(sentiment)

    if not coach_data:
        return _create_empty_bar_chart(f"No coach data available for version {version}")

    coaches = []
    avg_actions = []
    avg_pass_rates = []
    meeting_counts = []
    positive_sentiment_pct = []

    for coach, data in coach_data.items():
        # Include all coaches with meetings, not just those with actions
        if data["meetings"] > 0:
            coaches.append(coach)
            avg_actions.append(sum(data["actions"]) / len(data["actions"]) if data["actions"] else 0)
            avg_pass_rates.append(sum(data["pass_rates"]) / len(data["pass_rates"]) if data["pass_rates"] else 0)
            meeting_counts.append(data["meetings"])

            # Calculate positive sentiment percentage
            positive_count = sum(1 for s in data["sentiments"] if s.lower() == "positive")
            sentiment_pct = (positive_count / len(data["sentiments"])) * 100 if data["sentiments"] else 0
            positive_sentiment_pct.append(sentiment_pct)

    if not coaches:
        return _create_empty_bar_chart(f"No coach performance data for version {version}")

    df = pd.DataFrame({
        "coach": coaches,
        "avg_actions": avg_actions,
        "avg_pass_rate": avg_pass_rates,
        "meeting_count": meeting_counts,
        "positive_sentiment_pct": positive_sentiment_pct
    })

    # Create a multi-metric chart
    fig = px.bar(df, x="coach", y="meeting_count",
                title=f"Coach Performance Overview - Version {version}",
                hover_data={
                    "avg_actions": ":.1f",
                    "avg_pass_rate": ":.1f",
                    "positive_sentiment_pct": ":.1f"
                },
                labels={
                    "meeting_count": "Total Meetings",
                    "coach": "Coach",
                    "avg_actions": "Avg Actions/Meeting",
                    "avg_pass_rate": "Avg Pass Rate %",
                    "positive_sentiment_pct": "Positive Sentiment %"
                })

    return fig

def _create_sentiment_chart(items, version):
    """Create sentiment analysis chart"""
    sentiment_counts = {}
    for item in items:
        sentiment = item.get("sentiment", "Unknown")
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

    if not sentiment_counts:
        return _create_empty_pie_chart(f"No sentiment data for version {version}")

    df = pd.DataFrame({
        "sentiment": list(sentiment_counts.keys()),
        "count": list(sentiment_counts.values())
    })

    fig = px.pie(df, values="count", names="sentiment",
                title=f"Sentiment Distribution - Version {version}")
    return fig

def _create_themes_chart(items, version):
    """Create themes trend chart with actual financial coaching themes"""
    theme_counts = {}

    # Financial coaching theme keywords
    theme_keywords = {
        "Budgeting & Spending": ["budget", "spending", "expenses", "bills", "cost", "afford"],
        "Debt Management": ["debt", "loan", "credit", "owe", "payment", "mortgage"],
        "Savings & Emergency Fund": ["savings", "save", "emergency", "fund", "rainy day"],
        "Investment & Pension": ["invest", "investment", "portfolio", "pension", "retirement", "ISA"],
        "Insurance & Protection": ["insurance", "protection", "cover", "life insurance", "health"],
        "Career & Income": ["job", "career", "salary", "promotion", "income", "work"],
        "Financial Goals": ["goal", "target", "plan", "objective", "future", "dream"],
        "Banking & Accounts": ["account", "bank", "banking", "overdraft", "balance"],
        "Tax & Benefits": ["tax", "benefits", "allowance", "HMRC", "rebate"],
        "Property & Mortgage": ["house", "property", "mortgage", "rent", "deposit", "home"]
    }

    for item in items:
        # If actual themes are available in the data, prioritize them
        themes_from_data = item.get("themes", [])
        if themes_from_data and isinstance(themes_from_data, list):
            for theme in themes_from_data:
                theme_name = theme.get("label", "Unknown") if isinstance(theme, dict) else str(theme)
                if theme_name and theme_name != "Unknown":
                    theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1
        else:
            # Extract themes from meeting context using keywords
            item_found_themes = set()

            # Look for themes in various fields
            searchable_text = ""
            if item.get("employerName"):
                searchable_text += item.get("employerName", "").lower() + " "

            # You could also search in summary text if available
            # searchable_text += item.get("summary", "").lower()

            # Match themes based on keywords
            for theme_name, keywords in theme_keywords.items():
                for keyword in keywords:
                    if keyword in searchable_text:
                        item_found_themes.add(theme_name)
                        break

            # If no specific themes found, categorize by employer type
            if not item_found_themes:
                employer = item.get("employerName", "").lower()
                if any(word in employer for word in ["bank", "financial", "money"]):
                    item_found_themes.add("Banking & Accounts")
                elif "insurance" in employer:
                    item_found_themes.add("Insurance & Protection")
                elif any(word in employer for word in ["pension", "retirement"]):
                    item_found_themes.add("Investment & Pension")
                else:
                    item_found_themes.add("General Financial Coaching")

            # Add found themes to counts
            for theme in item_found_themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

    if not theme_counts:
        return _create_empty_bar_chart(f"No theme data for version {version}")

    # Limit to top 10 themes
    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    df = pd.DataFrame({
        "theme": [x[0] for x in sorted_themes],
        "count": [x[1] for x in sorted_themes]
    })

    fig = px.bar(df, x="theme", y="count",
                title=f"Top Discussion Themes - Version {version}",
                labels={"theme": "Theme", "count": "Frequency"})
    fig.update_layout(xaxis_tickangle=45)
    return fig

def _create_risk_chart(items, version):
    """Create risk and vulnerability chart"""
    risk_data = {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0, "No Risk": 0}

    for item in items:
        # Determine risk level based on multiple factors
        risk_level = "No Risk"

        # Check sentiment
        sentiment = (item.get("sentiment") or "").lower()
        if sentiment == "negative":
            risk_level = "High Risk"
        elif sentiment == "neutral":
            risk_level = "Medium Risk"

        # Check case pass rate
        pass_rate = item.get("casePassRate")
        if pass_rate is not None and float(pass_rate) < 0.5:
            risk_level = "High Risk"
        elif pass_rate is not None and float(pass_rate) < 0.8:
            if risk_level == "No Risk":
                risk_level = "Medium Risk"

        # Check failure count
        fail_count = item.get("caseFailedCount")
        if fail_count is not None and int(fail_count) >= 3:
            risk_level = "High Risk"
        elif fail_count is not None and int(fail_count) >= 1:
            if risk_level == "No Risk":
                risk_level = "Low Risk"

        risk_data[risk_level] += 1

    df = pd.DataFrame({
        "risk_level": list(risk_data.keys()),
        "count": list(risk_data.values())
    })

    # Color mapping for risk levels
    color_map = {
        "High Risk": "#dc2626",
        "Medium Risk": "#f59e0b",
        "Low Risk": "#10b981",
        "No Risk": "#6b7280"
    }

    fig = px.bar(df, x="risk_level", y="count",
                title=f"Risk & Vulnerability Assessment - Version {version}",
                color="risk_level",
                color_discrete_map=color_map,
                labels={"risk_level": "Risk Level", "count": "Meeting Count"})

    return fig

def render_escalation_table(items: list) -> str:
    """Render a focused table for escalation candidates with detailed reasoning"""
    if not items:
        return "<p>✅ No meetings requiring immediate attention</p>"

    header = """
    <thead>
      <tr style="background-color:#fee2e2;">
        <th style="text-align:left;">Date</th>
        <th style="text-align:left;">Meeting ID</th>
        <th style="text-align:left;">Coach</th>
        <th style="text-align:left;">Employer</th>
        <th style="text-align:left;">Risk Indicators</th>
        <th style="text-align:left;">Evidence</th>
        <th style="text-align:left;">Severity</th>
        <th style="text-align:left;">Action Required</th>
      </tr>
    </thead>"""

    rows = []
    for item in items[:10]:  # Limit to 10 most recent
        updated = html.escape(str(item.get("updatedAt", ""))[:10])  # Just date
        mid = html.escape(str(item.get("meetingId", "")))
        coach = html.escape(str(item.get("coach") or item.get("coachName", "Unknown")))
        employer = html.escape(str(item.get("employerName", "Unknown")))

        # Detailed risk analysis with evidence
        risk_indicators = []
        evidence_details = []
        severity_score = 0

        # Sentiment Analysis
        sentiment = (item.get("sentiment") or "").lower()
        if sentiment == "negative":
            risk_indicators.append("🔴 Negative Client Sentiment")
            evidence_details.append(f"Client sentiment: {sentiment.title()}")
            severity_score += 3

        # Compliance Issues
        pass_rate = item.get("casePassRate")
        if pass_rate is not None:
            pass_pct = float(pass_rate) * 100
            if pass_pct < 50:
                risk_indicators.append("🚨 Critical Compliance Failure")
                evidence_details.append(f"Case pass rate: {pass_pct:.0f}% (Critical: <50%)")
                severity_score += 4
            elif pass_pct < 80:
                risk_indicators.append("⚠️ Low Compliance Score")
                evidence_details.append(f"Case pass rate: {pass_pct:.0f}% (Target: >80%)")
                severity_score += 2

        fail_count = item.get("caseFailedCount")
        if fail_count is not None and int(fail_count) > 0:
            if int(fail_count) >= 3:
                risk_indicators.append("🔴 Multiple Compliance Failures")
                evidence_details.append(f"Failed {fail_count} compliance checks")
                severity_score += 3
            else:
                risk_indicators.append("⚠️ Compliance Concerns")
                evidence_details.append(f"Failed {fail_count} compliance check(s)")
                severity_score += 1

        # Action Items Analysis
        action_count = item.get("actionCount", 0)
        if action_count is not None:
            if action_count == 0:
                risk_indicators.append("⚠️ No Action Items Generated")
                evidence_details.append("Zero action items suggest incomplete coaching")
                severity_score += 2
            elif action_count > 10:
                risk_indicators.append("⚠️ Excessive Action Items")
                evidence_details.append(f"{action_count} action items may overwhelm client")
                severity_score += 1

        # Review Status
        status = item.get("status", "").upper()
        if status == "IN_REVIEW":
            risk_indicators.append("🔍 Pending Human Review")
            evidence_details.append("Flagged for manual quality review")
            severity_score += 1
        elif status == "FAILED":
            risk_indicators.append("❌ Processing Failed")
            evidence_details.append("System unable to complete analysis")
            severity_score += 2

        # If no specific risks found but flagged as escalation
        if not risk_indicators:
            risk_indicators.append("⚠️ Flagged for Review")
            evidence_details.append("Meeting flagged by automated screening")
            severity_score += 1

        # Determine severity level
        if severity_score >= 6:
            severity = "CRITICAL"
            severity_chip = chip("CRITICAL", "#dc2626", "#ffffff")
            action_required = "Immediate manager review required"
        elif severity_score >= 3:
            severity = "HIGH"
            severity_chip = chip("HIGH", "#dc2626", "#ffffff")
            action_required = "Review within 24 hours"
        else:
            severity = "MEDIUM"
            severity_chip = chip("MEDIUM", "#10b981", "#ffffff")
            action_required = "Review when convenient"

        # Format risk indicators and evidence
        risk_text = "<br/>".join(risk_indicators[:3])  # Show top 3 risks
        evidence_text = "<br/>".join(evidence_details[:3])  # Show top 3 evidence points

        # Meeting ID for inspection
        meeting_id = item.get("meetingId", "")
        inspection_link = f'''
            <strong>{meeting_id}</strong><br/>
        ''' if meeting_id else "No ID available"

        rows.append(f"""
          <tr style="border-bottom:1px solid #fee2e2;">
            <td>{updated}</td>
            <td>{inspection_link}</td>
            <td>{coach}</td>
            <td>{employer}</td>
            <td style="max-width:200px;">{risk_text}</td>
            <td style="max-width:250px;font-size:12px;color:#6b7280;">{evidence_text}</td>
            <td>{severity_chip}</td>
            <td style="font-size:12px;">{action_required}</td>
          </tr>""")

    return f"""
    <table style="border-collapse:collapse; width:100%; font-family: ui-sans-serif, system-ui; font-size: 14px; border:1px solid #fee2e2;">
      {header}
      <tbody>{''.join(rows)}</tbody>
    </table>"""

def refresh_all(page, page_size, version):
    version = (version or S3_PREFIX)
    data = fetch_summaries_from_athena(version)
    if "error" in data:
        banner = f"""<div style="padding:10px;border:1px solid #f0c36d;background:#fff8e1;border-radius:8px">
        <strong>Couldn’t load summaries.</strong><br/>{html.escape(str(data.get('error')))} (status: {html.escape(str(data.get('status','n/a')))})
        </div>"""
        return [], 1, banner, json.dumps(data, indent=2), "Page 1 of 1 (0 items)", 0, "0.00", "0.0%", gr.update(value=None), gr.update(choices=[S3_PREFIX, "all"], value=version)

    items = data.get("items", [])
    total, avg_actions, avg_pass = _kpis_from_items(items)
    page_items, page, total_pages, total_count, page_size = _paginate(items, page, page_size)
    table_html = render_summary_table(page_items)
    page_label = f"Page {page} of {total_pages} ({total_count} items)"
    fig = None
    sdf = _sentiment_df(items)
    if not sdf.empty:
        fig = px.bar(sdf, x="sentiment", y="count", title="Meetings by Sentiment")

    versions = _extract_versions(items)
    if "all" not in versions:
        versions = versions + ["all"] if versions else [S3_PREFIX, "all"]
    if version not in versions:
        versions = [version] + [v for v in versions if v != version]

    return (
        items,
        page,
        table_html,
        json.dumps(page_items, indent=2),
        page_label,
        total_count,
        f"{avg_actions:.2f}",
        f"{avg_pass*100:.1f}%",
        gr.update(value=fig),
        gr.update(choices=versions, value=version),
    )

def get_case_for_meeting(meeting_id: str) -> str:
    if not meeting_id.strip():
        return json.dumps({"error": "Enter a Meeting ID"}, indent=2)
    try:
        r = requests.get(CASE_URL, params={"meetingId": meeting_id.strip()}, timeout=15)
        if r.status_code != 200:
            return json.dumps({"error": "Failed to fetch case URL", "status": r.status_code, "body": r.text}, indent=2)
        url = r.json().get("caseUrl")
        if not url:
            return json.dumps({"error": "No caseUrl for this meeting"}, indent=2)
        cj = requests.get(url, timeout=20)
        if cj.status_code != 200:
            return json.dumps({"error": "Failed to fetch case JSON", "status": cj.status_code}, indent=2)
        try:
            case_data = cj.json()

            # Check if this is the new direct case check format
            if "results" in case_data and "overall" in case_data and "meeting_id" in case_data:
                # New format: case check data is directly in the response
                return json.dumps(case_data, indent=2)

            # Legacy format: extract case_json from inputContent
            input_content = case_data.get("inputContent", {})
            case_json_str = input_content.get("case_json")

            if case_json_str:
                # Parse the case_json string and return it formatted
                case_json = json.loads(case_json_str)
                return json.dumps(case_json, indent=2)
            else:
                # Neither format found, return the raw response for debugging
                return json.dumps({
                    "error": "No case_json found in case check data",
                    "available_keys": list(case_data.keys()),
                    "raw_response": case_data
                }, indent=2)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Failed to parse case_json: {str(e)}"}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to process case data: {str(e)}"}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

def get_summary_for_meeting(meeting_id: str) -> str:
    if not meeting_id.strip():
        return json.dumps({"error": "Enter a Meeting ID"}, indent=2)
    try:
        s = requests.get(STATUS_URL, params={"meetingId": meeting_id.strip()}, timeout=10)
        if s.status_code != 200:
            return json.dumps({"error": "Failed to fetch status", "status": s.status_code}, indent=2)
        data = s.json()
        if (data.get("status","")).upper() != "COMPLETED":
            return json.dumps({"error": f"Job not completed. Status={data.get('status')}"}, indent=2)
        url = data.get("downloadUrl")
        if not url:
            return json.dumps({"error": "No downloadUrl in status response."}, indent=2)
        jr = requests.get(url, timeout=20)
        if jr.status_code != 200:
            return json.dumps({"error": "Failed to fetch summary JSON", "status": jr.status_code}, indent=2)
        try:
            return json.dumps(jr.json(), indent=2)
        except Exception:
            return jr.text
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# =========================================================
# Athena helpers (version-aware with graceful fallback)
# =========================================================
def _athena_conn():
    if not PYATHENA_OK or not ATHENA_S3_STAGING:
        raise RuntimeError("PyAthena not configured. Set ATHENA_* env vars.")
    return _athena_connect(
        region_name=ATHENA_REGION,
        work_group=ATHENA_WORKGROUP,
        s3_staging_dir=ATHENA_S3_STAGING,
        cursor_class=PandasCursor,
    )

def run_df(sql: str) -> pd.DataFrame:
    with _athena_conn() as conn:
        return pd.read_sql_query(sql, conn)

# Sanitise version literal to avoid SQL injection
_VER_SAFE = re.compile(r"^[A-Za-z0-9/_=\.\-]+$")


# Base queries (assume views expose 'version'; we’ll try filtered first, then fallback)
Q_OVERVIEW_FILTERED = """
SELECT
  overall_sentiment AS sentiment,
  COUNT(DISTINCT meeting_id) AS meetings
FROM {db}.latest_summaries
WHERE version = '{ver}'
  AND year >= year(current_date) - 1
GROUP BY overall_sentiment
ORDER BY meetings DESC
"""

Q_OVERVIEW_UNFILTERED = """
SELECT
  overall_sentiment AS sentiment,
  COUNT(DISTINCT meeting_id) AS meetings
FROM {db}.latest_summaries
WHERE year >= year(current_date) - 1
GROUP BY overall_sentiment
ORDER BY meetings DESC
"""

Q_THEMES_FILTERED = """
SELECT
  t.label AS theme_label,
  COUNT(*) AS mentions
FROM {db}.latest_summaries s
CROSS JOIN UNNEST(s.themes) AS u(t)
WHERE s.version = '{ver}'
  AND s.year >= year(current_date) - 1
  AND t.label IS NOT NULL
GROUP BY t.label
ORDER BY mentions DESC
LIMIT 10
"""

Q_THEMES_UNFILTERED = """
SELECT
  t.label AS theme_label,
  COUNT(*) AS mentions
FROM {db}.latest_summaries s
CROSS JOIN UNNEST(s.themes) AS u(t)
WHERE s.year >= year(current_date) - 1
  AND t.label IS NOT NULL
GROUP BY t.label
ORDER BY mentions DESC
LIMIT 10
"""

Q_CASE_QUALITY_FILTERED = """
SELECT
  coach_name AS coach,
  AVG(quality_score) AS avg_quality_score,
  COUNT(*) AS meeting_count
FROM {db}.latest_summaries
WHERE version = '{ver}'
  AND coach_name IS NOT NULL
  AND year >= year(current_date) - 1
GROUP BY coach_name
ORDER BY avg_quality_score DESC
"""

Q_CASE_QUALITY_UNFILTERED = """
SELECT
  coach_name AS coach,
  AVG(quality_score) AS avg_quality_score,
  COUNT(*) AS meeting_count
FROM {db}.latest_summaries
WHERE coach_name IS NOT NULL
  AND year >= year(current_date) - 1
GROUP BY coach_name
ORDER BY avg_quality_score DESC
"""


def _try_versioned_sql(sql_filtered: str, sql_unfiltered: str, version: str):
    sql_f = sql_filtered.format(db=ATHENA_DATABASE, ver="1.2")
    sql_u = sql_unfiltered.format(db=ATHENA_DATABASE)
    try:
        df = run_df(sql_f)
        info = f"Filtered by version: `{version}`"
        return df, info
    except Exception as e:
        # Skip broken tables with JSON errors
        if "HIVE_CURSOR_ERROR" in str(e) or "JSONException" in str(e):
            print(f"⚠️  Skipping broken table due to JSON error")
            return pd.DataFrame(), "Table unavailable due to JSON formatting issues"

        # Fallback when views don't yet expose `version`
        try:
            df = run_df(sql_u)
            info = "Showing all versions (your Athena views likely lack a `version` column)."
            return df, info
        except Exception as e2:
            if "HIVE_CURSOR_ERROR" in str(e2) or "JSONException" in str(e2):
                print(f"⚠️  Both tables broken due to JSON error")
                return pd.DataFrame(), "Table unavailable due to JSON formatting issues"
            raise

def load_overview():
    try:
        df = run_df(Q_OVERVIEW_FILTERED.format(db=ATHENA_DATABASE, ver="1.2"))
        fig = px.bar(df, x="sentiment", y="meetings", title="Meetings by Sentiment (Athena)")
        return df, fig
    except Exception as e:
        try:
            df = run_df(Q_OVERVIEW_UNFILTERED.format(db=ATHENA_DATABASE))
            fig = px.bar(df, x="sentiment", y="meetings", title="Meetings by Sentiment (Athena)")
            return df, fig
        except Exception:
            msg = f"Athena error: {e}"
            return pd.DataFrame({"error":[msg]}), None

def load_themes():
    try:
        df = run_df(Q_THEMES_FILTERED.format(db=ATHENA_DATABASE, ver="1.2"))
        fig = px.bar(df, x="theme_label", y="mentions", title="Top Themes")
        return df, fig
    except Exception as e:
        try:
            df = run_df(Q_THEMES_UNFILTERED.format(db=ATHENA_DATABASE))
            fig = px.bar(df, x="theme_label", y="mentions", title="Top Themes")
            return df, fig
        except Exception:
            msg = f"Athena error: {e}"
            return pd.DataFrame({"error":[msg]}), None

def load_case_quality():
    try:
        df = run_df(Q_CASE_QUALITY_FILTERED.format(db=ATHENA_DATABASE, ver="1.2"))
        if df is not None and not df.empty:
            # Update for new column names from partitioned tables
            if "avg_quality_score" in df.columns:
                df["avg_quality_score_pct"] = (df["avg_quality_score"].astype(float) * 100).round(2)
                fig = px.bar(df, x="coach", y="avg_quality_score", title="Avg Quality Score by Coach")
            elif "avg_case_pass" in df.columns:
                df["avg_case_pass_pct"] = (df["avg_case_pass"].astype(float) * 100).round(2)
                fig = px.bar(df, x="coach", y="avg_case_pass_pct", title="Avg Case Pass Rate by Coach (%)")
            else:
                fig = None
        else:
            fig = None
        return df, fig
    except Exception as e:
        try:
            df = run_df(Q_CASE_QUALITY_UNFILTERED.format(db=ATHENA_DATABASE))
            if df is not None and not df.empty:
                if "avg_quality_score" in df.columns:
                    df["avg_quality_score_pct"] = (df["avg_quality_score"].astype(float) * 100).round(2)
                    fig = px.bar(df, x="coach", y="avg_quality_score", title="Avg Quality Score by Coach")
                elif "avg_case_pass" in df.columns:
                    df["avg_case_pass_pct"] = (df["avg_case_pass"].astype(float) * 100).round(2)
                    fig = px.bar(df, x="coach", y="avg_case_pass_pct", title="Avg Case Pass Rate by Coach (%)")
                else:
                    fig = None
            else:
                fig = None
            return df, fig
        except Exception:
            msg = f"Athena error: {e}"
            return pd.DataFrame({"error":[msg]}), None

# Version comparison queries
Q_VERSION_COMPARISON = """
SELECT
  version,
  model_version,
  prompt_version,
  COUNT(*) as summary_count,
  AVG(sentiment_confidence) as avg_sentiment_confidence,
  AVG(action_count) as avg_action_count,
  AVG(theme_count) as avg_theme_count
FROM {db}.version_comparison
WHERE year >= year(current_date) - 1
GROUP BY version, model_version, prompt_version
ORDER BY version DESC, model_version, prompt_version
"""

Q_MONTHLY_TRENDS = """
SELECT
  year,
  month,
  version,
  summary_count,
  unique_meetings,
  active_coaches,
  avg_actions,
  avg_themes,
  avg_sentiment_confidence,
  positive_meetings,
  negative_meetings,
  neutral_meetings
FROM {db}.monthly_trends
WHERE year >= year(current_date) - 1
ORDER BY year DESC, month DESC, version
"""

def load_version_comparison():
    """Load version comparison analytics"""
    try:
        if not PYATHENA_OK:
            return pd.DataFrame(), px.bar(), "Athena not configured. Install PyAthena and set environment variables."

        sql = Q_VERSION_COMPARISON.format(db=ATHENA_DATABASE)
        df = run_df(sql)

        if df.empty:
            return df, px.bar(), "No version comparison data available. Run the Athena DDL first."

        # Create comparison chart showing action count vs theme count by version
        fig = px.scatter(df,
                        x="avg_action_count",
                        y="avg_theme_count",
                        size="summary_count",
                        color="version",
                        hover_data=["model_version", "prompt_version", "avg_sentiment_confidence"],
                        title="Version Performance: Action Count vs Theme Count",
                        labels={
                            "avg_action_count": "Avg Action Items per Meeting",
                            "avg_theme_count": "Avg Themes per Meeting"
                        })

        info = f"Comparing {len(df)} version configurations. Bubble size = summary count."
        return df, fig, info

    except Exception as e:
        return pd.DataFrame(), px.bar(), f"Error loading version comparison: {e}"

def load_monthly_trends():
    """Load monthly performance trends"""
    try:
        if not PYATHENA_OK:
            return pd.DataFrame(), px.bar(), "Athena not configured. Install PyAthena and set environment variables."

        sql = Q_MONTHLY_TRENDS.format(db=ATHENA_DATABASE)
        df = run_df(sql)

        if df.empty:
            return df, px.line(), "No monthly trends data available. Run the Athena DDL first."

        # Create month-year column for better plotting
        df['month_year'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)

        # Create multi-line chart showing key metrics over time by version
        fig = px.line(df,
                     x="month_year",
                     y="avg_sentiment_confidence",
                     color="version",
                     title="Monthly Sentiment Confidence Trends by Version",
                     labels={
                         "month_year": "Month",
                         "avg_sentiment_confidence": "Avg Sentiment Confidence"
                     })

        # Add secondary metrics as hover data
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>" +
                         "Month: %{x}<br>" +
                         "Sentiment Confidence: %{y:.2f}<br>" +
                         "Summaries: %{customdata[0]}<br>" +
                         "Avg Actions: %{customdata[1]:.1f}<br>" +
                         "Avg Themes: %{customdata[2]:.1f}<br>" +
                         "<extra></extra>",
            customdata=df[['summary_count', 'avg_actions', 'avg_themes']].values
        )

        info = f"Monthly trends for {df['year'].nunique()} years, {df['version'].nunique()} versions. Hover for details."
        return df, fig, info

    except Exception as e:
        return pd.DataFrame(), px.line(), f"Error loading monthly trends: {e}"

def load_coach_analytics():
    """Load coach performance analytics"""
    try:
        if not PYATHENA_OK:
            return pd.DataFrame(), px.bar(), "Athena not configured. Install PyAthena and set environment variables."

        sql = f"SELECT * FROM {ATHENA_DATABASE}.coach_analytics WHERE year >= year(current_date) - 1 ORDER BY coach_name, year DESC, month DESC"
        df = run_df(sql)

        if df.empty:
            return df, px.bar(), "No coach analytics data available. Run the Athena DDL first."

        # Create bar chart showing meeting count by coach and version
        fig = px.bar(df,
                    x="coach_name",
                    y="meeting_count",
                    color="version",
                    title="Meeting Count by Coach and Version",
                    labels={
                        "coach_name": "Coach Name",
                        "meeting_count": "Meeting Count"
                    },
                    hover_data=["avg_actions_per_meeting", "avg_themes_per_meeting", "avg_sentiment_confidence"])

        fig.update_layout(xaxis_tickangle=-45)

        info = f"Coach analytics for {df['coach_name'].nunique()} coaches across {df['version'].nunique()} versions."
        return df, fig, info

    except Exception as e:
        return pd.DataFrame(), px.bar(), f"Error loading coach analytics: {e}"

def load_performance_metrics():
    """Load performance metrics"""
    try:
        if not PYATHENA_OK:
            return pd.DataFrame(), px.bar(), "Athena not configured. Install PyAthena and set environment variables."

        sql = f"SELECT * FROM {ATHENA_DATABASE}.performance_metrics WHERE year >= year(current_date) - 1 ORDER BY year DESC, month DESC, version"
        df = run_df(sql)

        if df.empty:
            return df, px.bar(), "No performance metrics data available. Run the Athena DDL first."

        # Create a grouped bar chart showing metrics by version
        fig = px.bar(df,
                    x="version",
                    y="total_summaries",
                    color="version",
                    title="Total Summaries by Version",
                    labels={
                        "version": "Version",
                        "total_summaries": "Total Summaries"
                    },
                    hover_data=["unique_meetings", "unique_coaches", "avg_actions", "avg_themes"])

        info = f"Performance metrics across {df['version'].nunique()} versions, {df['total_summaries'].sum()} total summaries."
        return df, fig, info

    except Exception as e:
        return pd.DataFrame(), px.bar(), f"Error loading performance metrics: {e}"

# =========================================================
# UI
# =========================================================
with gr.Blocks(title="Octopus Money — Summariser & Case Checks") as app:
    gr.Markdown("### 🐙 Octopus Money — Coaching Session Summariser")
    gr.Markdown("Provide *either* a transcript **or** a Zoom Meeting ID. Optionally enable case checking for AI quality analysis. Use the Dashboard tab for case checks; use Insights for Athena charts.")

    # ---------- Submit ----------
    with gr.Tab("Submit"):
        with gr.Row():
            meeting_id   = gr.Textbox(label="OM Meeting ID", placeholder="om-2025-08-28-abc")
            coach_name   = gr.Textbox(label="Coach Name (optional)",     placeholder="e.g., John Doe")
            employer_name= gr.Textbox(label="Employer Name (optional)",  placeholder="e.g., Santander")
        transcript      = gr.Textbox(label="Transcript (paste text) — optional", lines=8)
        zoom_meeting_id = gr.Textbox(label="Zoom Meeting ID — optional", placeholder="95682401830")

        with gr.Row():
            enable_case_check = gr.Checkbox(label="Enable Case Checking", value=False, info="Run AI case quality analysis (requires A2I flow)")

        case_check_warning = gr.Markdown(f"⚠️ **Note:** Case checking enabled - will add additional processing time and requires human review. [Access A2I Portal]({A2I_PORTAL_URL}) to review case checks.", visible=False)

        run_btn         = gr.Button("Generate Summary")

        result_json     = gr.Code(label="Summary JSON", language="json", value="")

        # Interactions
        run_btn.click(submit_and_get_json, inputs=[meeting_id, coach_name, employer_name, transcript, zoom_meeting_id, enable_case_check], outputs=[result_json])
        enable_case_check.change(lambda x: gr.update(visible=x), inputs=[enable_case_check], outputs=[case_check_warning])

        # Inspect existing meetings section
        gr.Markdown("---")
        gr.Markdown("### 🔍 Inspect Existing Meeting")
        gr.Markdown("Look up previously processed meetings to review summaries and case check results.")

        with gr.Row():
            mid_in = gr.Textbox(label="Meeting ID", placeholder="om-2025-08-28-abc", info="Enter the Meeting ID to inspect")
            with gr.Column():
                get_summary_btn = gr.Button("📄 Get Summary JSON", variant="secondary")
                get_case_btn = gr.Button("⚖️ Get Case Check JSON", variant="secondary")

        with gr.Row():
            summary_out = gr.Code(label="Summary JSON", language="json", lines=10)
            case_out = gr.Code(label="Case Check JSON", language="json", lines=10)

        get_summary_btn.click(fn=get_summary_for_meeting, inputs=[mid_in], outputs=[summary_out])
        get_case_btn.click(fn=get_case_for_meeting, inputs=[mid_in], outputs=[case_out])

    # ---------- Dashboard (API) ----------
    with gr.Tab("Dashboard"):
        gr.Markdown("### 📊 Coaching Insights Dashboard")
        gr.Markdown("🎯 **Showing v1.2 data with complete compliance fields**")
        gr.Markdown("💡 **Note**: Only v1.2 data is supported. All summaries use the latest schema version.")

        # Refresh button
        with gr.Row():
            refresh_dashboard_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")

        # Data quality indicator
        data_quality_info = gr.Markdown("")

        # Quick KPIs that matter
        with gr.Row():
            kpi_total = gr.Number(label="Total meetings", value=0, interactive=False)
            kpi_avg_actions = gr.Textbox(label="Avg actions / meeting", value="0.00", interactive=False)
            kpi_escalations = gr.Number(label="🚨 Escalation candidates", value=0, interactive=False)
            kpi_negative_sentiment = gr.Number(label="😞 Negative sentiment", value=0, interactive=False)

        # Version-specific analytics charts
        gr.Markdown("### 📈 Performance Analytics")

        with gr.Row():
            coach_performance_plot = gr.Plot(label="Coach Performance")
            sentiment_plot = gr.Plot(label="Sentiment Analysis")

        with gr.Row():
            themes_plot = gr.Plot(label="Theme Trends")
            risk_plot = gr.Plot(label="Risk & Vulnerability")

        # Load basic KPIs with data filtering
        def load_dashboard_kpis():
            version = "summaries/v=1.2"
            data = fetch_summaries_from_athena(version)
            if "error" in data:
                return 0, "0.00", 0, 0, ""

            raw_items = data.get("items", [])
            # Filter to only include complete records
            items = _filter_complete_records(raw_items, version)

            # Create data quality info
            total_raw = len(raw_items)
            total_filtered = len(items)
            filtered_out = total_raw - total_filtered

            if version == "summaries/v=1.2" and filtered_out > 0:
                quality_info = f"📊 **Data Quality**: Showing {total_filtered} complete records, filtered out {filtered_out} incomplete records"
            elif filtered_out > 0:
                quality_info = f"📊 **Data Quality**: Showing {total_filtered} of {total_raw} records (some may have missing compliance data)"
            else:
                quality_info = f"✅ **Data Quality**: All {total_filtered} records have complete data"

            total, avg_actions, _ = _kpis_from_items(items)
            escalations = sum(1 for item in items if _is_escalation_candidate(item))
            negative_sentiment = sum(1 for item in items if (item.get("sentiment") or "").lower() == "negative")

            return total, f"{avg_actions:.2f}", escalations, negative_sentiment, quality_info

        # Load initial charts and KPIs
        def load_all_dashboard_data():
            """Load all dashboard data (v1.2 only)"""
            # Load KPIs
            kpis = load_dashboard_kpis()

            # Load charts
            charts = load_version_specific_analytics()

            return (*kpis, *charts)

        app.load(
            fn=lambda: load_all_dashboard_data(),
            inputs=None,
            outputs=[kpi_total, kpi_avg_actions, kpi_escalations, kpi_negative_sentiment, data_quality_info,
                    coach_performance_plot, sentiment_plot, themes_plot, risk_plot]
        )

        # Wire up refresh button
        refresh_dashboard_btn.click(
            fn=lambda: load_all_dashboard_data(),
            inputs=None,
            outputs=[kpi_total, kpi_avg_actions, kpi_escalations, kpi_negative_sentiment, data_quality_info,
                    coach_performance_plot, sentiment_plot, themes_plot, risk_plot]
        )


        # Escalation Summary Section
        gr.Markdown("### 🚨 Meetings Requiring Attention")
        escalation_refresh_btn = gr.Button("Refresh Escalations")
        escalation_table = gr.HTML()

        # Load escalation data with filtering
        def load_escalations():
            version = "summaries/v=1.2"
            data = fetch_summaries_from_athena(version)
            if "error" in data:
                return f"<p>Error loading escalation data for version {version}</p>"

            raw_items = data.get("items", [])
            # Filter to only include complete records
            items = _filter_complete_records(raw_items, version)

            escalation_items = [item for item in items if _is_escalation_candidate(item)]

            if not escalation_items:
                return f"<p>✅ No meetings requiring immediate attention in version {version}</p>"

            return render_escalation_table(escalation_items)

        app.load(fn=lambda: load_escalations(), inputs=None, outputs=[escalation_table])
        escalation_refresh_btn.click(fn=load_escalations, inputs=None, outputs=[escalation_table])


    # ---------- Insights (Athena) ----------
    with gr.Tab("Insights"):
        if not PYATHENA_OK or not ATHENA_S3_STAGING:
            gr.Markdown("> ⚠️ Set `ATHENA_S3_STAGING`, `ATHENA_REGION`, `ATHENA_WORKGROUP`, `ATHENA_DATABASE` env vars to enable this tab.")
        else:
            gr.Markdown("### 📈 Business Insights")

            # --- Top Discussion Themes (Business Value)
            gr.Markdown("**What are clients talking about?**")
            th_plot = gr.Plot()

            def load_themes_fallback():
                try:
                    # Try Athena first
                    df, fig = load_themes()
                    if fig and not df.empty:
                        return fig
                    else:
                        # Fallback to API data
                        return load_themes_from_api()
                except Exception as e:
                    # Fallback to API data
                    return load_themes_from_api()

            app.load(lambda: load_themes_fallback(), None, [th_plot])

            # --- Coach Quality Performance (Business Value)
            gr.Markdown("**Coach performance by quality score**")
            cq_plot = gr.Plot()

            def load_quality_fallback():
                try:
                    # Try Athena first
                    df, fig = load_case_quality()
                    if fig and not df.empty:
                        return fig
                    else:
                        # Fallback to API data
                        return load_quality_from_api()
                except Exception as e:
                    # Fallback to API data
                    return load_quality_from_api()

            app.load(lambda: load_quality_fallback(), None, [cq_plot])

if __name__ == "__main__":
    # Lambda Web Adapter expects 0.0.0.0 and PORT
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "8080")))
