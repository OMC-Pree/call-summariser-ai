"""
Model prediction loader - fetches predictions from S3 case check results.
"""
import json
from pathlib import Path
from typing import List, Optional, Dict
import sys
import os

# Add summariser to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "summariser"))

from utils.aws_clients import AWSClients
from constants import SUMMARY_BUCKET, S3_PREFIX, SCHEMA_VERSION, CASE_CHECK_SCHEMA_VERSION
from evals.models import ModelPrediction, BinaryLabel


def load_case_check_from_s3(meeting_id: str, year: int, month: int) -> Optional[Dict]:
    """
    Load case check JSON from S3 for a specific meeting.

    Args:
        meeting_id: Meeting ID (without .mp3 extension)
        year: Year for partitioned path
        month: Month for partitioned path

    Returns:
        Case check JSON dict, or None if not found
    """
    s3 = AWSClients.s3()

    # Try partitioned path first (current format)
    key = (
        f"{S3_PREFIX}/supplementary/version={SCHEMA_VERSION}/"
        f"year={year}/month={month:02d}/meeting_id={meeting_id}/"
        f"case_check.v{CASE_CHECK_SCHEMA_VERSION}.json"
    )

    try:
        response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except s3.exceptions.NoSuchKey:
        # Try non-partitioned fallback
        key_fallback = f"{S3_PREFIX}/{year:04d}/{month:02d}/{meeting_id}/case_check.v{CASE_CHECK_SCHEMA_VERSION}.json"
        try:
            response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=key_fallback)
            return json.loads(response['Body'].read().decode('utf-8'))
        except s3.exceptions.NoSuchKey:
            return None


def extract_vulnerability_prediction(case_check_data: Dict) -> ModelPrediction:
    """
    Extract vulnerability detection prediction from case check JSON.

    The case check contains a "vulnerability_identified" check with status:
    - "Fail" = vulnerability detected (needs attention)
    - "Competent" = no vulnerability or handled appropriately
    - Other statuses mapped conservatively

    Args:
        case_check_data: Case check JSON dict from S3

    Returns:
        ModelPrediction with binary label and confidence

    Raises:
        ValueError: If case check data is invalid or missing required fields
    """
    meeting_id = case_check_data.get("meeting_id")
    if not meeting_id:
        raise ValueError("Case check data missing 'meeting_id'")

    results = case_check_data.get("results", [])
    if not results:
        raise ValueError(f"Case check data for {meeting_id} has no results")

    # Find vulnerability_identified check
    vuln_check = None
    for result in results:
        if result.get("id") == "vulnerability_identified":
            vuln_check = result
            break

    if not vuln_check:
        raise ValueError(
            f"Case check for {meeting_id} missing 'vulnerability_identified' check. "
            f"Available checks: {[r.get('id') for r in results]}"
        )

    status = vuln_check.get("status", "")
    confidence = float(vuln_check.get("confidence", 0.0))
    evidence_quote = vuln_check.get("evidence_quote", "")
    comment = vuln_check.get("comment", "")

    # Map status to binary label
    # "Fail" means vulnerability was identified and NOT handled appropriately
    # "Competent" means no vulnerability OR vulnerability handled appropriately
    # Conservative mapping: treat uncertain cases as not_vulnerable
    if status == "Fail":
        # Vulnerability detected and NOT handled properly = vulnerable
        label: BinaryLabel = "vulnerable"
        reason = f"Status: {status}. {comment}"
    elif status == "Competent":
        # No vulnerability OR handled appropriately = not_vulnerable
        label = "not_vulnerable"
        reason = f"Status: {status}. {comment}"
    elif status == "CompetentWithDevelopment":
        # Vulnerability handled but could be better = not_vulnerable (was addressed)
        label = "not_vulnerable"
        reason = f"Status: {status}. {comment}"
    elif status == "NotApplicable":
        # Check not applicable = not_vulnerable
        label = "not_vulnerable"
        reason = f"Status: {status}. {comment}"
    elif status == "Inconclusive":
        # Unclear = not_vulnerable (conservative, avoid false positives)
        label = "not_vulnerable"
        reason = f"Status: {status} (inconclusive, mapped to not_vulnerable). {comment}"
    else:
        raise ValueError(f"Unknown status '{status}' for vulnerability check in {meeting_id}")

    return ModelPrediction(
        meeting_id=meeting_id,
        model_vulnerability_label=label,
        confidence=confidence,
        model_reason=reason,
        check_id="vulnerability_identified",
        status=status,
        evidence_quote=evidence_quote
    )


def load_predictions_from_s3(
    meeting_ids: List[str],
    year: int,
    month: int
) -> List[ModelPrediction]:
    """
    Load model predictions from S3 for a list of meeting IDs.

    Args:
        meeting_ids: List of meeting IDs (without .mp3 extension)
        year: Year for partitioned path
        month: Month for partitioned path

    Returns:
        List of ModelPrediction objects (only for meetings with case checks)

    Note:
        Meetings without case checks in S3 will be skipped with a warning.
        This allows partial evaluation when not all meetings have been processed.
    """
    predictions = []
    missing = []

    for meeting_id in meeting_ids:
        # Clean meeting_id (remove .mp3 if present)
        clean_id = meeting_id.replace(".mp3", "")

        case_check = load_case_check_from_s3(clean_id, year, month)

        if case_check is None:
            missing.append(clean_id)
            continue

        try:
            prediction = extract_vulnerability_prediction(case_check)
            predictions.append(prediction)
        except ValueError as e:
            print(f"Warning: Failed to extract prediction for {clean_id}: {e}")
            missing.append(clean_id)

    if missing:
        print(f"Warning: {len(missing)} meetings missing case checks in S3: {missing[:5]}...")

    return predictions


def load_predictions_from_csv(csv_path: str | Path) -> List[ModelPrediction]:
    """
    Load model predictions from CSV file.

    CSV must have columns:
    - meeting_id
    - model_vulnerability_label (vulnerable/not_vulnerable)
    - confidence (0.0-1.0)
    - model_reason (optional)

    Args:
        csv_path: Path to CSV with predictions

    Returns:
        List of ModelPrediction objects

    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If CSV has invalid format
    """
    import csv

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")

    predictions = []
    required_cols = {"meeting_id", "model_vulnerability_label", "confidence"}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"CSV has no headers: {csv_path}")

        normalized_headers = {col.strip().lower(): col for col in reader.fieldnames}
        missing_cols = required_cols - set(normalized_headers.keys())
        if missing_cols:
            raise ValueError(
                f"Predictions CSV missing columns: {missing_cols}\n"
                f"Found columns: {list(reader.fieldnames)}"
            )

        col_map = {norm: orig for norm, orig in normalized_headers.items()}

        for row_num, row in enumerate(reader, start=2):
            if not any(row.values()):
                continue

            meeting_id = row.get(col_map["meeting_id"], "").strip()
            if not meeting_id:
                continue

            try:
                prediction = ModelPrediction(
                    meeting_id=meeting_id.replace(".mp3", ""),
                    model_vulnerability_label=row.get(col_map["model_vulnerability_label"], "").strip(),
                    confidence=float(row.get(col_map["confidence"], "0.0")),
                    model_reason=row.get(col_map.get("model_reason", ""), "").strip()
                )
                predictions.append(prediction)
            except Exception as e:
                raise ValueError(
                    f"Error parsing row {row_num} in {csv_path.name}: {e}\n"
                    f"Row: {row}"
                ) from e

    if not predictions:
        raise ValueError(f"No valid predictions found in CSV: {csv_path}")

    return predictions
