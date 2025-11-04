"""
Evaluation scoring: compute metrics and create evaluation records.
"""
from typing import List, Tuple, Dict
from collections import defaultdict

from evals.models import (
    AggregatedGroundTruth,
    ModelPrediction,
    EvaluationRecord,
    EvaluationMetrics
)


def match_predictions_to_ground_truth(
    ground_truth: List[AggregatedGroundTruth],
    predictions: List[ModelPrediction]
) -> Tuple[List[EvaluationRecord], List[str]]:
    """
    Match predictions to ground truth records and create evaluation records.

    Args:
        ground_truth: List of aggregated ground truth records
        predictions: List of model predictions

    Returns:
        Tuple of:
        - List of EvaluationRecord objects (one per matched meeting)
        - List of meeting_ids that had no prediction (warnings)

    Raises:
        ValueError: If no predictions match any ground truth records
    """
    # Check for duplicate meeting_ids in predictions
    seen_ids = set()
    duplicates = []
    for p in predictions:
        if p.meeting_id in seen_ids:
            duplicates.append(p.meeting_id)
        seen_ids.add(p.meeting_id)

    if duplicates:
        raise ValueError(f"Duplicate prediction meeting_ids found: {duplicates}")

    # Index predictions by meeting_id for fast lookup
    pred_by_id: Dict[str, ModelPrediction] = {
        p.meeting_id: p for p in predictions
    }

    evaluation_records = []
    missing_predictions = []

    for gt in ground_truth:
        prediction = pred_by_id.get(gt.meeting_id)

        if prediction is None:
            # No prediction for this ground truth record
            missing_predictions.append(gt.meeting_id)
            continue

        # Create evaluation record
        eval_record = EvaluationRecord.from_ground_truth_and_prediction(
            ground_truth=gt,
            prediction=prediction,
            notes=""
        )
        evaluation_records.append(eval_record)

    if not evaluation_records:
        raise ValueError(
            "No predictions matched ground truth records. "
            "Check that meeting_ids align between ground truth CSV and predictions."
        )

    return evaluation_records, missing_predictions


def compute_metrics(
    evaluation_records: List[EvaluationRecord],
    run_timestamp: str
) -> EvaluationMetrics:
    """
    Compute evaluation metrics from evaluation records.

    Args:
        evaluation_records: List of evaluation records
        run_timestamp: Timestamp for this evaluation run

    Returns:
        EvaluationMetrics with precision, recall, F1, confusion matrix

    Raises:
        ValueError: If evaluation_records is empty
    """
    if not evaluation_records:
        raise ValueError("Cannot compute metrics from empty evaluation records")

    return EvaluationMetrics.from_results(evaluation_records, run_timestamp)


def print_metrics_summary(metrics: EvaluationMetrics) -> None:
    """
    Print a formatted summary of evaluation metrics to console.

    Args:
        metrics: Evaluation metrics to display
    """
    print("\n" + "=" * 60)
    print(f"EVALUATION METRICS - {metrics.run_timestamp}")
    print("=" * 60)
    print(f"\nTotal Samples: {metrics.total_samples}")
    print(f"  - Vulnerable: {metrics.total_vulnerable}")
    print(f"  - Not Vulnerable: {metrics.total_not_vulnerable}")

    print("\nConfusion Matrix:")
    print(f"  True Positives  (TP): {metrics.true_positives:3d}")
    print(f"  True Negatives  (TN): {metrics.true_negatives:3d}")
    print(f"  False Positives (FP): {metrics.false_positives:3d}")
    print(f"  False Negatives (FN): {metrics.false_negatives:3d}")

    print("\nPerformance Metrics:")
    print(f"  Precision: {metrics.precision:.4f} ({metrics.precision*100:.2f}%)")
    print(f"  Recall:    {metrics.recall:.4f} ({metrics.recall*100:.2f}%)")
    print(f"  F1 Score:  {metrics.f1_score:.4f} ({metrics.f1_score*100:.2f}%)")
    print(f"  Accuracy:  {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
    print("=" * 60 + "\n")


def analyze_errors(
    evaluation_records: List[EvaluationRecord]
) -> Dict[str, List[EvaluationRecord]]:
    """
    Group evaluation records by result type for error analysis.

    Args:
        evaluation_records: List of evaluation records

    Returns:
        Dict mapping result type (TP/TN/FP/FN) to list of records
    """
    grouped: Dict[str, List[EvaluationRecord]] = defaultdict(list)

    for record in evaluation_records:
        grouped[record.result].append(record)

    return dict(grouped)


def print_error_analysis(
    evaluation_records: List[EvaluationRecord],
    show_limit: int = 5
) -> None:
    """
    Print detailed error analysis showing false positives and false negatives.

    Args:
        evaluation_records: List of evaluation records
        show_limit: Maximum number of examples to show per error type
    """
    grouped = analyze_errors(evaluation_records)

    # False Positives (predicted vulnerable, actually not)
    fps = grouped.get("FP", [])
    if fps:
        print(f"\n{'='*60}")
        print(f"FALSE POSITIVES ({len(fps)} total) - Model said vulnerable, actually not")
        print(f"{'='*60}")
        for i, record in enumerate(fps[:show_limit], 1):
            print(f"\n{i}. Meeting: {record.meeting_id}")
            print(f"   Adviser: {record.adviser_name}")
            print(f"   Ground Truth: {record.expected_vulnerability_label} ({record.max_vulnerability_rating})")
            print(f"   Model Predicted: {record.model_vulnerability_label} (confidence: {record.confidence:.2f})")
            print(f"   Reason: {record.model_reason[:200]}")
        if len(fps) > show_limit:
            print(f"\n   ... and {len(fps) - show_limit} more FPs")

    # False Negatives (predicted not vulnerable, actually vulnerable)
    fns = grouped.get("FN", [])
    if fns:
        print(f"\n{'='*60}")
        print(f"FALSE NEGATIVES ({len(fns)} total) - Model missed vulnerabilities")
        print(f"{'='*60}")
        for i, record in enumerate(fns[:show_limit], 1):
            print(f"\n{i}. Meeting: {record.meeting_id}")
            print(f"   Adviser: {record.adviser_name}")
            print(f"   Ground Truth: {record.expected_vulnerability_label} ({record.max_vulnerability_rating})")
            print(f"   Vulnerability Types: {', '.join(record.vulnerability_types)}")
            print(f"   Model Predicted: {record.model_vulnerability_label} (confidence: {record.confidence:.2f})")
            print(f"   Reason: {record.model_reason[:200]}")
            if record.evidence_quotes:
                print(f"   Evidence: {record.evidence_quotes[0][:150]}...")
        if len(fns) > show_limit:
            print(f"\n   ... and {len(fns) - show_limit} more FNs")

    print(f"\n{'='*60}\n")
