"""
S3 Partitioning Utilities for Athena Optimization

Creates properly partitioned S3 paths for efficient Athena querying.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional


class S3Partitioner:
    """Creates Athena-optimized S3 paths with proper partitioning"""

    def __init__(self, base_prefix: str = "summaries"):
        self.base_prefix = base_prefix.rstrip('/')
        self.athena_partitioned = os.getenv('ATHENA_PARTITIONED', 'true').lower() == 'true'
        self.schema_version = "1.2"  # Only support v1.2

    def get_summary_path(self,
                        meeting_id: str,
                        timestamp: Optional[datetime] = None,
                        is_latest: bool = True) -> str:
        """
        Generate partitioned S3 path for summary storage

        Format: summaries/version=1.1/year=2025/month=09/meeting_id=123/summary.json
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Always use Athena-optimized partitioned structure (v1.2 only)
        path_parts = [
            self.base_prefix,
            f"version={self.schema_version}",
            f"year={timestamp.year}",
            f"month={timestamp.month:02d}",
            f"meeting_id={meeting_id}",
            "summary.json" if is_latest else f"summary_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
        ]

        return "/".join(path_parts)

    def get_transcript_path(self, meeting_id: str, timestamp: Optional[datetime] = None) -> str:
        """Generate partitioned path for transcript storage"""
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Always use partitioned structure (v1.2 only)
        path_parts = [
            "transcripts",
            f"version={self.schema_version}",
            f"year={timestamp.year}",
            f"month={timestamp.month:02d}",
            f"meeting_id={meeting_id}",
            "transcript.json"
        ]

        return "/".join(path_parts)

    def get_athena_table_location(self) -> str:
        """Get S3 location for Athena table definition"""
        return f"s3://{os.getenv('SUMMARY_BUCKET', 'your-bucket')}/{self.base_prefix}/"

    def create_athena_ddl(self, bucket_name: str) -> str:
        """
        Generate Athena DDL for creating partitioned table
        """
        return f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {os.getenv('ATHENA_DATABASE', 'call_summaries')}.summaries (
    meeting_id string,
    summary_text string,
    key_points array<string>,
    action_items array<string>,
    participants array<string>,
    duration_minutes int,
    sentiment_score double,
    processing_time_seconds double,
    model_version string,
    prompt_version string,
    created_at timestamp,
    metadata struct<
        coach_name: string,
        meeting_type: string,
        quality_score: double,
        word_count: int
    >
)
PARTITIONED BY (
    version string,
    year int,
    month int
)
STORED AS JSON
LOCATION 's3://{bucket_name}/{self.base_prefix}/'
TBLPROPERTIES (
    'has_encrypted_data'='false',
    'projection.enabled'='true',
    'projection.version.type'='enum',
    'projection.version.values'='1.2',
    'projection.year.type'='integer',
    'projection.year.range'='2024,2030',
    'projection.year.interval'='1',
    'projection.month.type'='integer',
    'projection.month.range'='1,12',
    'projection.month.interval'='1',
    'projection.month.digits'='2',
    'storage.location.template'='s3://{bucket_name}/{self.base_prefix}/version=${{version}}/year=${{year}}/month=${{month}}/'
);
"""

    def get_partition_info(self, meeting_id: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Get partition information for metadata indexing"""
        if timestamp is None:
            timestamp = datetime.utcnow()

        return {
            'version': self.schema_version,
            'year': timestamp.year,
            'month': timestamp.month,
            'day': timestamp.day,
            'meeting_id': meeting_id,
            'timestamp': timestamp.isoformat(),
            'partition_path': f"version={self.schema_version}/year={timestamp.year}/month={timestamp.month:02d}"
        }

    def migrate_legacy_path_to_partitioned(self, legacy_path: str) -> str:
        """
        Convert legacy S3 paths to partitioned format

        From: summaries/2025/08/meeting-id/summary.v1.1.json
        To:   summaries/version=1.1/year=2025/month=08/meeting_id=meeting-id/summary.json
        """
        import re

        # Match legacy format: summaries/YYYY/MM/meeting-id/filename
        legacy_match = re.match(r'summaries/(\d{4})/(\d{2})/([^/]+)/(.+)', legacy_path)
        if legacy_match:
            year, month, meeting_id, filename = legacy_match.groups()

            # Extract version from filename if present
            version_match = re.search(r'v(\d+\.\d+)', filename)
            version = version_match.group(1) if version_match else self.schema_version

            return f"summaries/version={version}/year={year}/month={month}/meeting_id={meeting_id}/summary.json"

        # Already in partitioned format or different format
        return legacy_path


def get_s3_partitioner() -> S3Partitioner:
    """Factory function to get configured S3 partitioner"""
    base_prefix = os.getenv('S3_PREFIX', 'summaries')
    return S3Partitioner(base_prefix)