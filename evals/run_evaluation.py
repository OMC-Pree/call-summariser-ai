#!/usr/bin/env python3
"""
Vulnerability Detection Evaluation Pipeline

Usage:
    python evals/run_evaluation.py \
        --ground-truth evals/golden_data/vulnerability_ground_truth.csv \
        --year 2025 \
        --month 10 \
        --source s3

    python evals/run_evaluation.py \
        --ground-truth evals/golden_data/vulnerability_ground_truth.csv \
        --predictions evals/golden_data/predictions.csv \
        --source csv

Outputs:
    - evals/runs/{timestamp}/evaluation_results.csv
    - evals/runs/{timestamp}/metrics_summary.json
    - evals/runs/{timestamp}/aggregated_ground_truth.csv (for inspection)
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import csv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.data_loader import (
    load_ground_truth_csv,
    aggregate_by_meeting,
    save_aggregated_csv
)
from evals.prediction_loader import (
    load_predictions_from_s3,
    load_predictions_from_csv
)
from evals.scorer import (
    match_predictions_to_ground_truth,
    compute_metrics,
    print_metrics_summary,
    print_error_analysis
)


def create_run_directory() -> Path:
    """
    Create timestamped run directory for this evaluation.

    Returns:
        Path to run directory (evals/runs/{timestamp}/)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_evaluation_results(
    evaluation_records,
    output_path: Path
) -> None:
    """
    Save evaluation results to CSV with all details.

    Args:
        evaluation_records: List of EvaluationRecord objects
        output_path: Path to save CSV
    """
    if not evaluation_records:
        raise ValueError("Cannot save empty evaluation results")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(evaluation_records[0].model_dump().keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in evaluation_records:
            row = record.model_dump()
            # Convert lists to readable format
            row["vulnerability_types"] = "; ".join(row["vulnerability_types"])
            row["evidence_quotes"] = " || ".join(row["evidence_quotes"])
            writer.writerow(row)


def save_metrics_summary(
    metrics,
    output_path: Path
) -> None:
    """
    Save metrics summary to JSON.

    Args:
        metrics: EvaluationMetrics object
        output_path: Path to save JSON
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics.model_dump(), f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run vulnerability detection evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to ground truth CSV (exported from Google Sheets)"
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["s3", "csv"],
        help="Source for model predictions: 's3' (from case checks) or 'csv' (from file)"
    )

    # S3 source arguments
    parser.add_argument(
        "--year",
        type=int,
        help="Year for S3 predictions (required if --source s3)"
    )
    parser.add_argument(
        "--month",
        type=int,
        help="Month for S3 predictions (required if --source s3)"
    )

    # CSV source arguments
    parser.add_argument(
        "--predictions",
        help="Path to predictions CSV (required if --source csv)"
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        help="Custom output directory (default: evals/runs/{timestamp}/)"
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=5,
        help="Number of error examples to show (default: 5)"
    )

    args = parser.parse_args()

    # Validate source-specific arguments
    if args.source == "s3" and (args.year is None or args.month is None):
        parser.error("--year and --month are required when --source is 's3'")
    if args.source == "csv" and args.predictions is None:
        parser.error("--predictions is required when --source is 'csv'")

    print("\n" + "=" * 60)
    print("VULNERABILITY DETECTION EVALUATION PIPELINE")
    print("=" * 60)

    # Step 1: Load and clean ground truth
    print(f"\n1. Loading ground truth from: {args.ground_truth}")
    ground_truth_records = load_ground_truth_csv(args.ground_truth)
    print(f"   Loaded {len(ground_truth_records)} ground truth records")

    # Step 2: Aggregate by meeting
    print("\n2. Aggregating by meeting_id...")
    aggregated_gt = aggregate_by_meeting(ground_truth_records)
    print(f"   Aggregated to {len(aggregated_gt)} unique meetings")
    print(f"   - Vulnerable (High/Critical): {sum(1 for gt in aggregated_gt if gt.expected_vulnerability_label == 'vulnerable')}")
    print(f"   - Not Vulnerable (Low/Medium): {sum(1 for gt in aggregated_gt if gt.expected_vulnerability_label == 'not_vulnerable')}")

    # Step 3: Load predictions
    print(f"\n3. Loading predictions from {args.source}...")
    if args.source == "s3":
        meeting_ids = [gt.meeting_id for gt in aggregated_gt]
        predictions = load_predictions_from_s3(meeting_ids, args.year, args.month)
    else:  # csv
        predictions = load_predictions_from_csv(args.predictions)

    print(f"   Loaded {len(predictions)} predictions")
    print(f"   - Predicted Vulnerable: {sum(1 for p in predictions if p.model_vulnerability_label == 'vulnerable')}")
    print(f"   - Predicted Not Vulnerable: {sum(1 for p in predictions if p.model_vulnerability_label == 'not_vulnerable')}")

    # Step 4: Match and create evaluation records
    print("\n4. Matching predictions to ground truth...")
    evaluation_records, missing_preds = match_predictions_to_ground_truth(
        aggregated_gt, predictions
    )
    print(f"   Matched {len(evaluation_records)} records")
    if missing_preds:
        print(f"   Warning: {len(missing_preds)} meetings had no predictions")
        if len(missing_preds) <= 10:
            print(f"   Missing: {missing_preds}")
        else:
            print(f"   Missing (first 10): {missing_preds[:10]}")

    # Step 5: Compute metrics
    print("\n5. Computing evaluation metrics...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = compute_metrics(evaluation_records, timestamp)

    # Step 6: Save results
    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_directory()

    print(f"\n6. Saving results to: {run_dir}")

    # Save evaluation results CSV
    results_path = run_dir / "evaluation_results.csv"
    save_evaluation_results(evaluation_records, results_path)
    print(f"   ✓ Saved evaluation results: {results_path}")

    # Save metrics summary JSON
    metrics_path = run_dir / "metrics_summary.json"
    save_metrics_summary(metrics, metrics_path)
    print(f"   ✓ Saved metrics summary: {metrics_path}")

    # Save aggregated ground truth for inspection
    aggregated_path = run_dir / "aggregated_ground_truth.csv"
    save_aggregated_csv(aggregated_gt, aggregated_path)
    print(f"   ✓ Saved aggregated ground truth: {aggregated_path}")

    # Save configuration for reproducibility
    config = {
        "run_timestamp": timestamp,
        "ground_truth_file": str(args.ground_truth),
        "prediction_source": args.source,
        "total_ground_truth_records": len(ground_truth_records),
        "total_meetings": len(aggregated_gt),
        "total_predictions": len(predictions),
        "total_matched": len(evaluation_records),
        "total_missing_predictions": len(missing_preds)
    }
    if args.source == "s3":
        config["year"] = args.year
        config["month"] = args.month
    else:
        config["predictions_file"] = str(args.predictions)

    config_path = run_dir / "run_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved run configuration: {config_path}")

    # Step 7: Display results
    print_metrics_summary(metrics)

    if args.show_errors > 0:
        print_error_analysis(evaluation_records, show_limit=args.show_errors)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {run_dir}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    exit(main())
