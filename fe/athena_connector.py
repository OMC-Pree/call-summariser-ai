#!/usr/bin/env python3

import os
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Athena configuration
ATHENA_REGION = os.getenv("ATHENA_REGION", "eu-west-2")
ATHENA_WORKGROUP = os.getenv("ATHENA_WORKGROUP", "primary")
ATHENA_S3_STAGING = os.getenv("ATHENA_S3_STAGING")
ATHENA_DATABASE = os.getenv("ATHENA_DATABASE", "call_summaries")
S3_PREFIX = os.getenv("S3_PREFIX", "summaries")

# Try to import PyAthena
PYATHENA_OK = True
try:
    from pyathena import connect as _athena_connect
    from pyathena.pandas.cursor import PandasCursor
except ImportError:
    PYATHENA_OK = False

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

def _ver(v: str) -> str:
    v = v or S3_PREFIX
    return v if _VER_SAFE.match(v) else "summaries"

def fetch_summaries_from_athena(version: str) -> dict:
    """Fetch summaries data from Athena for the specified version"""
    try:
        if not PYATHENA_OK or not ATHENA_S3_STAGING:
            return {"error": "Athena not configured. Set ATHENA_* env vars."}

        # First, let's check what tables are available
        show_tables_sql = f"SHOW TABLES IN {ATHENA_DATABASE}"
        print(f"🔍 Checking available tables...")

        try:
            tables_df = run_df(show_tables_sql)
            print("Available tables:")
            for _, row in tables_df.iterrows():
                print(f"  {row['tab_name']}")
        except Exception as e:
            print(f"⚠️  Could not show tables: {e}")

        # First check what versions are available
        version_sql = f"""
        SELECT DISTINCT version, COUNT(*) as count
        FROM {ATHENA_DATABASE}.summaries
        GROUP BY version
        ORDER BY count DESC
        """

        print(f"🔍 Checking available versions...")
        try:
            version_df = run_df(version_sql)
            print("Available versions:")
            for _, row in version_df.iterrows():
                print(f"  {row['version']}: {row['count']} items")
        except Exception as e:
            print(f"⚠️  Could not check versions: {e}")

        # Let's also check unique meeting IDs to see the actual meeting count
        meeting_sql = f"""
        SELECT version, COUNT(DISTINCT meeting_id) as unique_meetings, COUNT(*) as total_records
        FROM {ATHENA_DATABASE}.summaries
        GROUP BY version
        ORDER BY total_records DESC
        """

        print(f"🔍 Checking unique meetings vs total records...")
        try:
            meeting_df = run_df(meeting_sql)
            print("Meeting counts by version:")
            for _, row in meeting_df.iterrows():
                print(f"  {row['version']}: {row['unique_meetings']} unique meetings, {row['total_records']} total records")
        except Exception as e:
            print(f"⚠️  Could not check meeting counts: {e}")

        # Let's analyze the data quality issue - 3085 records for 150 meetings
        print(f"🔍 Analyzing data quality issues...")

        # Check if there are actual unique meeting IDs we can identify
        try:
            # Look for columns that might contain meeting identifiers
            sample_sql = f"""
            SELECT *
            FROM {ATHENA_DATABASE}.summaries
            WHERE version = '1.1'
            LIMIT 5
            """

            sample_df = run_df(sample_sql)
            print(f"Sample columns: {list(sample_df.columns)}")

            # Look at actual data content to understand the duplication
            if not sample_df.empty:
                print("\nFirst row content inspection:")
                first_row = sample_df.iloc[0]
                for col in sample_df.columns:
                    val = str(first_row[col])
                    if val and val != 'nan' and len(val) > 0:
                        print(f"  {col}: {val[:200]}...")

        except Exception as e:
            print(f"⚠️  Could not analyze data: {e}")

        # Check for potential meeting ID patterns in the data
        try:
            # Try to find actual meeting IDs or summary content
            content_sql = f"""
            SELECT version, summary, meeting, call_metadata
            FROM {ATHENA_DATABASE}.summaries
            WHERE version = '1.1'
            AND (summary IS NOT NULL OR meeting IS NOT NULL)
            LIMIT 3
            """

            content_df = run_df(content_sql)
            print(f"\nContent sample shape: {content_df.shape}")

            if not content_df.empty:
                print("Non-null content found:")
                for idx, row in content_df.iterrows():
                    print(f"  Row {idx}:")
                    for col in ['summary', 'meeting', 'call_metadata']:
                        val = str(row[col])
                        if val and val != 'nan':
                            print(f"    {col}: {val[:100]}...")

        except Exception as e:
            print(f"⚠️  Could not check content: {e}")

        # Now check if the issue is in how we're counting vs actual unique data
        try:
            # Try to identify unique processing runs or meeting instances
            count_sql = f"""
            SELECT
                version,
                COUNT(*) as total_records,
                COUNT(DISTINCT summary) as unique_summaries,
                COUNT(DISTINCT meeting) as unique_meetings,
                COUNT(DISTINCT call_metadata) as unique_metadata
            FROM {ATHENA_DATABASE}.summaries
            WHERE version IN ('1.1', '1.0')
            GROUP BY version
            """

            count_df = run_df(count_sql)
            print(f"\nData uniqueness analysis:")
            for _, row in count_df.iterrows():
                print(f"  Version {row['version']}:")
                print(f"    Total records: {row['total_records']}")
                print(f"    Unique summaries: {row['unique_summaries']}")
                print(f"    Unique meetings: {row['unique_meetings']}")
                print(f"    Unique metadata: {row['unique_metadata']}")

        except Exception as e:
            print(f"⚠️  Could not analyze uniqueness: {e}")

        # Create a clean query that extracts unique meetings and parses JSON fields
        sql = f"""
        SELECT DISTINCT
            summary_schema_version,
            model_version,
            prompt_version,
            meeting,
            sentiment,
            themes,
            summary,
            actions,
            call_metadata,
            insights,
            version,
            year,
            month
        FROM {ATHENA_DATABASE}.summaries
        WHERE version = '{_ver(version)}'
        AND meeting IS NOT NULL
        AND summary IS NOT NULL
        AND meeting NOT LIKE '%concurrent%'  -- Exclude test data
        ORDER BY year DESC, month DESC
        LIMIT 200
        """

        print(f"🔍 Executing clean Athena query for version: {_ver(version)}")
        df = run_df(sql)

        # Convert DataFrame to the format expected by the dashboard
        items = []
        for _, row in df.iterrows():
            try:
                # Parse the JSON fields safely
                import json
                import re

                # Extract meeting ID from meeting JSON
                meeting_str = str(row.get("meeting", "{}"))
                meeting_id_match = re.search(r'"?id"?\s*[:=]\s*"?([^",}]+)"?', meeting_str)
                meeting_id = meeting_id_match.group(1) if meeting_id_match else None

                # Extract coach name
                coach_match = re.search(r'"?coach"?\s*[:=]\s*"?([^",}]+)"?', meeting_str)
                coach_name = coach_match.group(1) if coach_match else None

                # Extract employer name
                employer_match = re.search(r'"?employername"?\s*[:=]\s*"?([^",}]+)"?', meeting_str)
                employer_name = employer_match.group(1) if employer_match else None

                # Parse sentiment
                sentiment_str = str(row.get("sentiment", "{}"))
                sentiment_match = re.search(r'"?label"?\s*[:=]\s*"?([^",}]+)"?', sentiment_str)
                sentiment = sentiment_match.group(1) if sentiment_match else "Unknown"

                # Parse insights for action count
                insights_str = str(row.get("insights", "{}"))
                action_count_match = re.search(r'"?action_count"?\s*[:=]\s*"?([^",}]+)"?', insights_str)
                action_count = action_count_match.group(1) if action_count_match else 0

                # Parse escalation flag
                escalation_match = re.search(r'"?is_escalation_candidate"?\s*[:=]\s*(true|false)', insights_str)
                is_escalation = escalation_match.group(1) == "true" if escalation_match else False

                # Skip if no meaningful data
                if not meeting_id or not coach_name:
                    continue

                item = {
                    "meetingId": meeting_id,
                    "coachName": coach_name,
                    "employerName": employer_name or "Unknown",
                    "sentiment": sentiment,
                    "updatedAt": f"{row.get('year', 2025)}-{row.get('month', 9):02d}-01",
                    "actionCount": int(action_count) if str(action_count).isdigit() else 0,
                    "isEscalation": is_escalation,
                    "summary": str(row.get("summary", ""))[:500],  # Limit summary length
                    "version": row.get("version"),
                    "themes": str(row.get("themes", ""))[:200],
                    # For v=1.2 compatibility (will be None for v=1.1)
                    "casePassRate": None,
                    "caseFailedCount": None,
                    "qualityScore": None,
                    "riskLevel": None,
                    "vulnerabilityScore": None,
                    "severityLevel": None,
                    "caseCheckEnabled": False
                }
                items.append(item)

            except Exception as e:
                print(f"⚠️  Error parsing row: {e}")
                continue

        print(f"✅ Extracted {len(items)} clean meeting records")
        return {"items": items}

    except Exception as e:
        return {"error": f"Athena query failed: {str(e)}"}

if __name__ == "__main__":
    # Test the function
    print("Testing Athena integration with clean data extraction...")
    print(f"ATHENA_DATABASE: {ATHENA_DATABASE}")
    print(f"ATHENA_S3_STAGING: {ATHENA_S3_STAGING}")

    if not PYATHENA_OK:
        print("❌ PyAthena not available")
        exit(1)

    if not ATHENA_S3_STAGING:
        print("❌ ATHENA_S3_STAGING not configured")
        exit(1)

    print("✅ Athena configuration looks good")

    # Test with v=1.1 first to verify our data cleaning works
    print("\n🔍 Testing clean data extraction for v=1.1...")
    result = fetch_summaries_from_athena("summaries/v=1.1")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        items = result.get("items", [])
        print(f"✅ Successfully extracted {len(items)} clean items from Athena")

        if items:
            print("\nSample cleaned item:")
            sample = items[0]
            for key, value in sample.items():
                print(f"  {key}: {value}")

            # Show coach distribution
            coaches = {}
            for item in items:
                coach = item.get("coachName", "Unknown")
                coaches[coach] = coaches.get(coach, 0) + 1

            print(f"\nCoach distribution ({len(items)} total meetings):")
            for coach, count in sorted(coaches.items(), key=lambda x: x[1], reverse=True):
                print(f"  {coach}: {count} meetings")

        else:
            print("⚠️  No items found in v=1.1 data")

    # Test fetching v=1.2 data
    print("\n🔍 Testing fetch_summaries_from_athena for v=1.2...")
    result = fetch_summaries_from_athena("summaries/v=1.2")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        items = result.get("items", [])
        print(f"✅ Successfully fetched {len(items)} items from Athena for v=1.2")

        if items:
            print("\nv=1.2 sample item:")
            sample = items[0]
            for key, value in sample.items():
                print(f"  {key}: {value}")
        else:
            print("⚠️  No items found in v=1.2 data")