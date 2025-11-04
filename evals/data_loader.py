"""
Data loading and cleaning for vulnerability detection evaluation.
"""
import csv
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from evals.models import GroundTruthRecord, AggregatedGroundTruth


def load_ground_truth_csv(csv_path: str | Path) -> List[GroundTruthRecord]:
    """
    Load and validate ground truth data from CSV export.

    Args:
        csv_path: Path to CSV file exported from Google Sheets

    Returns:
        List of validated GroundTruthRecord objects

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV has invalid format or missing required columns
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    records = []
    skipped_rows = []
    required_cols = {
        "meeting_id", "adviser_name", "client_email",
        "call_date", "call_type", "vulnerability_rating",
        "vulnerability_type", "evidence_quote"
    }

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Validate headers
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no headers: {csv_path}")

        # Normalize column names (strip whitespace, lowercase)
        normalized_headers = {col.strip().lower(): col for col in reader.fieldnames}

        missing_cols = required_cols - set(normalized_headers.keys())
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}\n"
                f"Found columns: {list(reader.fieldnames)}"
            )

        # Create mapping from normalized to actual column names
        col_map = {norm: orig for norm, orig in normalized_headers.items()}

        for row_num, row in enumerate(reader, start=2):  # Start at 2 (1=header)
            # Skip empty rows
            if not any(row.values()):
                continue

            # Skip rows without meeting_id (incomplete data)
            meeting_id_raw = row.get(col_map["meeting_id"], "").strip()
            if not meeting_id_raw:
                continue

            try:
                # Clean and validate row
                # Note: meeting_id is the zoom_id (numeric identifier)
                cleaned_row = {
                    "meeting_id": meeting_id_raw,
                    "adviser_name": row.get(col_map["adviser_name"], "").strip(),
                    "client_email": row.get(col_map["client_email"], "").strip(),
                    "call_date": row.get(col_map["call_date"], "").strip(),
                    "call_type": row.get(col_map["call_type"], "").strip(),
                    "vulnerability_rating": row.get(col_map["vulnerability_rating"], "").strip(),
                    "vulnerability_type": row.get(col_map["vulnerability_type"], "").strip(),
                    "evidence_quote": row.get(col_map["evidence_quote"], "").strip()
                }

                # Validate with Pydantic
                record = GroundTruthRecord(**cleaned_row)
                records.append(record)

            except ValueError as e:
                # Skip rows with invalid meeting_id (non-numeric zoom ID)
                error_msg = str(e)
                if "meeting_id" in error_msg.lower():
                    skipped_rows.append({
                        "row": row_num,
                        "meeting_id": meeting_id_raw,
                        "reason": "Invalid meeting_id (non-numeric)"
                    })
                    print(f"⚠️  Skipping row {row_num}: Invalid meeting_id '{meeting_id_raw}' (must be numeric zoom ID)")
                    continue
                else:
                    # Re-raise other validation errors
                    raise ValueError(
                        f"Error parsing row {row_num} in {csv_path.name}: {e}\n"
                        f"Row data: {row}"
                    ) from e
            except Exception as e:
                raise ValueError(
                    f"Error parsing row {row_num} in {csv_path.name}: {e}\n"
                    f"Row data: {row}"
                ) from e

    # Report skipped rows summary
    if skipped_rows:
        print(f"\n⚠️  Skipped {len(skipped_rows)} rows with invalid meeting_id:")
        for skip in skipped_rows[:5]:  # Show first 5
            print(f"   Row {skip['row']}: '{skip['meeting_id']}' (must be numeric zoom ID)")
        if len(skipped_rows) > 5:
            print(f"   ... and {len(skipped_rows) - 5} more\n")

    if not records:
        raise ValueError(f"No valid records found in CSV: {csv_path}")

    return records


def aggregate_by_meeting(records: List[GroundTruthRecord]) -> List[AggregatedGroundTruth]:
    """
    Aggregate multiple vulnerability instances per meeting into one row.

    Each meeting may have multiple rows in the CSV (multiple vulnerabilities detected).
    This function groups by meeting_id and creates one aggregated record per meeting
    with the maximum vulnerability rating and all evidence quotes.

    Args:
        records: List of ground truth records

    Returns:
        List of aggregated records (one per unique meeting_id)
    """
    # Group records by meeting_id
    grouped: Dict[str, List[GroundTruthRecord]] = defaultdict(list)
    for record in records:
        grouped[record.meeting_id].append(record)

    # Aggregate each group
    aggregated = []
    for meeting_id, meeting_records in grouped.items():
        agg = AggregatedGroundTruth.from_records(meeting_records)
        aggregated.append(agg)

    # Sort by meeting_id for consistent ordering
    aggregated.sort(key=lambda x: x.meeting_id)

    return aggregated


def save_aggregated_csv(
    aggregated: List[AggregatedGroundTruth],
    output_path: str | Path
) -> None:
    """
    Save aggregated ground truth to CSV for inspection.

    Args:
        aggregated: List of aggregated records
        output_path: Path to save CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not aggregated:
        raise ValueError("Cannot save empty aggregated data")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Get field names from first record
        fieldnames = list(aggregated[0].model_dump().keys())

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in aggregated:
            row = record.model_dump()
            # Convert lists to semicolon-separated strings for CSV
            row["vulnerability_types"] = "; ".join(row["vulnerability_types"])
            row["evidence_quotes"] = " || ".join(row["evidence_quotes"])
            writer.writerow(row)
