"""
Data models for vulnerability detection evaluation pipeline.
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# Ground truth vulnerability ratings
# Supports: None/0, Marginal/1, Low/2, Medium/3, High/4, Critical/5
VulnerabilityRating = Literal["Critical/5", "High/4", "Medium/3", "Low/2", "Marginal/1", "None/0"]

# Binary classification labels
BinaryLabel = Literal["vulnerable", "not_vulnerable"]

# Evaluation result status
EvalResult = Literal["TP", "TN", "FP", "FN"]


class GroundTruthRecord(BaseModel):
    """Single row from Google Sheets (may be multiple rows per meeting)

    Note: meeting_id is the zoom_id (numeric identifier)
    """
    meeting_id: str  # This is the zoom_id (numeric)
    adviser_name: str
    client_email: str
    call_date: str
    call_type: str
    vulnerability_rating: VulnerabilityRating
    vulnerability_type: Optional[str] = ""
    evidence_quote: Optional[str] = ""

    @field_validator('meeting_id', mode='before')
    @classmethod
    def validate_and_clean_meeting_id(cls, v):
        """
        Validate and clean meeting_id (which is the zoom_id):
        - Accepts string or numeric types
        - Must be numeric (spaces and hyphens are OK in strings)
        - Removes spaces and hyphens for consistency
        - Example: "935 9850 2685" → "93598502685"

        Raises ValueError if meeting_id contains non-numeric characters
        (e.g., "Does not have recording link in harbour")
        """
        if not v:
            raise ValueError(f"meeting_id must be non-empty, got: {v}")

        # Convert numeric types to string
        if isinstance(v, (int, float)):
            v = str(int(v))
        elif not isinstance(v, str):
            raise ValueError(f"meeting_id must be a string or numeric, got: {type(v)}")

        # Clean the meeting_id (remove spaces and hyphens)
        cleaned = v.replace(' ', '').replace('-', '')

        # Check if it's all digits
        if not cleaned.isdigit():
            raise ValueError(
                f"Invalid meeting_id: '{v}'. Must be numeric zoom ID. "
                f"This record will be skipped."
            )

        # Return the cleaned version (no spaces/hyphens)
        return cleaned


class AggregatedGroundTruth(BaseModel):
    """One row per meeting with aggregated vulnerability label

    Note: meeting_id is the zoom_id (numeric identifier)
    """
    meeting_id: str  # This is the zoom_id (numeric)
    adviser_name: str
    client_email: str
    call_date: str
    call_type: str

    # Aggregated fields
    max_vulnerability_rating: VulnerabilityRating
    vulnerability_count: int = 0
    vulnerability_types: List[str] = Field(default_factory=list)
    evidence_quotes: List[str] = Field(default_factory=list)

    # Binary label (derived from max_vulnerability_rating)
    expected_vulnerability_label: BinaryLabel

    @classmethod
    def from_records(cls, records: List[GroundTruthRecord]) -> "AggregatedGroundTruth":
        """Aggregate multiple vulnerability instances for one meeting"""
        if not records:
            raise ValueError("Cannot aggregate empty record list")

        # All records should have same meeting_id
        meeting_ids = set(r.meeting_id for r in records)
        if len(meeting_ids) > 1:
            raise ValueError(f"Multiple meeting_ids in record group: {meeting_ids}")

        first = records[0]

        # Find maximum vulnerability rating
        rating_order = {
            "Critical/5": 5,
            "High/4": 4,
            "Medium/3": 3,
            "Low/2": 2,
            "Marginal/1": 1,
            "None/0": 0
        }
        max_rating = max(records, key=lambda r: rating_order[r.vulnerability_rating])

        # Collect all vulnerability types and evidence quotes
        vuln_types = [r.vulnerability_type for r in records if r.vulnerability_type]
        evidence = [r.evidence_quote for r in records if r.evidence_quote]

        # Derive binary label: High/4 or Critical/5 = vulnerable
        expected_label: BinaryLabel = (
            "vulnerable" if rating_order[max_rating.vulnerability_rating] >= 4
            else "not_vulnerable"
        )

        return cls(
            meeting_id=first.meeting_id,
            adviser_name=first.adviser_name,
            client_email=first.client_email,
            call_date=first.call_date,
            call_type=first.call_type,
            max_vulnerability_rating=max_rating.vulnerability_rating,
            vulnerability_count=len(records),
            vulnerability_types=vuln_types,
            evidence_quotes=evidence,
            expected_vulnerability_label=expected_label
        )


class ModelPrediction(BaseModel):
    """Model prediction for a single meeting"""
    meeting_id: str
    model_vulnerability_label: BinaryLabel
    confidence: float = Field(ge=0.0, le=1.0)
    model_reason: Optional[str] = ""

    # Optional: raw case check data
    check_id: Optional[str] = None  # e.g., "vulnerability_identified"
    status: Optional[str] = None    # e.g., "Competent", "Fail"
    evidence_quote: Optional[str] = None


class EvaluationRecord(BaseModel):
    """Single evaluation result combining ground truth + prediction

    Note: meeting_id is the zoom_id (numeric identifier)
    """
    meeting_id: str  # This is the zoom_id (numeric)
    adviser_name: str
    client_email: str
    call_date: str
    call_type: str

    # Ground truth
    expected_vulnerability_label: BinaryLabel
    max_vulnerability_rating: VulnerabilityRating
    vulnerability_count: int
    vulnerability_types: List[str] = Field(default_factory=list)
    evidence_quotes: List[str] = Field(default_factory=list)

    # Model prediction
    model_vulnerability_label: BinaryLabel
    confidence: float
    model_reason: Optional[str] = ""

    # Evaluation result
    result: EvalResult
    notes_for_improvement: Optional[str] = ""

    @classmethod
    def from_ground_truth_and_prediction(
        cls,
        ground_truth: AggregatedGroundTruth,
        prediction: ModelPrediction,
        notes: str = ""
    ) -> "EvaluationRecord":
        """Create evaluation record from ground truth and prediction"""
        # Determine TP/TN/FP/FN
        expected = ground_truth.expected_vulnerability_label
        predicted = prediction.model_vulnerability_label

        if expected == "vulnerable" and predicted == "vulnerable":
            result: EvalResult = "TP"
        elif expected == "not_vulnerable" and predicted == "not_vulnerable":
            result = "TN"
        elif expected == "not_vulnerable" and predicted == "vulnerable":
            result = "FP"
        else:  # expected == "vulnerable" and predicted == "not_vulnerable"
            result = "FN"

        return cls(
            meeting_id=ground_truth.meeting_id,
            adviser_name=ground_truth.adviser_name,
            client_email=ground_truth.client_email,
            call_date=ground_truth.call_date,
            call_type=ground_truth.call_type,
            expected_vulnerability_label=expected,
            max_vulnerability_rating=ground_truth.max_vulnerability_rating,
            vulnerability_count=ground_truth.vulnerability_count,
            vulnerability_types=ground_truth.vulnerability_types,
            evidence_quotes=ground_truth.evidence_quotes,
            model_vulnerability_label=predicted,
            confidence=prediction.confidence,
            model_reason=prediction.model_reason or "",
            result=result,
            notes_for_improvement=notes
        )


class EvaluationMetrics(BaseModel):
    """Summary metrics for evaluation run"""
    run_timestamp: str
    total_samples: int

    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # Derived metrics
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)

    # Class distribution
    total_vulnerable: int
    total_not_vulnerable: int

    @classmethod
    def from_results(cls, results: List[EvaluationRecord], run_timestamp: str) -> "EvaluationMetrics":
        """Calculate metrics from evaluation results"""
        if not results:
            raise ValueError("Cannot calculate metrics from empty results")

        # Count TP/TN/FP/FN
        tp = sum(1 for r in results if r.result == "TP")
        tn = sum(1 for r in results if r.result == "TN")
        fp = sum(1 for r in results if r.result == "FP")
        fn = sum(1 for r in results if r.result == "FN")

        # Calculate metrics with zero-division handling
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(results)

        # Class distribution
        total_vulnerable = sum(1 for r in results if r.expected_vulnerability_label == "vulnerable")
        total_not_vulnerable = len(results) - total_vulnerable

        return cls(
            run_timestamp=run_timestamp,
            total_samples=len(results),
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1_score, 4),
            accuracy=round(accuracy, 4),
            total_vulnerable=total_vulnerable,
            total_not_vulnerable=total_not_vulnerable
        )
