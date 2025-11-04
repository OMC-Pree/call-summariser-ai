"""
Tests for vulnerability detection evaluation pipeline.
"""
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.models import (
    GroundTruthRecord,
    AggregatedGroundTruth,
    ModelPrediction,
    EvaluationRecord,
    EvaluationMetrics
)
from evals.data_loader import load_ground_truth_csv, aggregate_by_meeting
from evals.scorer import match_predictions_to_ground_truth, compute_metrics


class TestGroundTruthRecord:
    """Test ground truth record parsing and cleaning"""

    def test_clean_meeting_id_removes_mp3_extension(self):
        """Meeting ID should have .mp3 extension removed"""
        record = GroundTruthRecord(
            meeting_id="123abc.mp3",
            adviser_name="Jane Smith",
            zoom_id="921 2562 6617",
            client_email="test@example.com",
            call_date="27/10/2025 11:01:00",
            call_type="Starter",
            vulnerability_rating="High/4",
            vulnerability_type="Health: Chronic Illness",
            evidence_quote="I have COPD"
        )
        assert record.meeting_id == "123abc"

    def test_clean_meeting_id_without_extension(self):
        """Meeting ID without extension should remain unchanged"""
        record = GroundTruthRecord(
            meeting_id="123abc",
            adviser_name="Jane Smith",
            zoom_id="921 2562 6617",
            client_email="test@example.com",
            call_date="27/10/2025 11:01:00",
            call_type="Starter",
            vulnerability_rating="High/4",
            vulnerability_type="Health: Chronic Illness",
            evidence_quote="I have COPD"
        )
        assert record.meeting_id == "123abc"


class TestAggregatedGroundTruth:
    """Test aggregation logic"""

    def test_aggregate_single_record(self):
        """Single record should aggregate correctly"""
        records = [
            GroundTruthRecord(
                meeting_id="123abc",
                adviser_name="Jane Smith",
                zoom_id="921 2562 6617",
                client_email="test@example.com",
                call_date="27/10/2025 11:01:00",
                call_type="Starter",
                vulnerability_rating="High/4",
                vulnerability_type="Health: Chronic Illness",
                evidence_quote="I have COPD"
            )
        ]

        agg = AggregatedGroundTruth.from_records(records)

        assert agg.meeting_id == "123abc"
        assert agg.max_vulnerability_rating == "High/4"
        assert agg.vulnerability_count == 1
        assert agg.expected_vulnerability_label == "vulnerable"
        assert "Health: Chronic Illness" in agg.vulnerability_types

    def test_aggregate_multiple_vulnerabilities(self):
        """Multiple vulnerabilities should take max rating"""
        records = [
            GroundTruthRecord(
                meeting_id="123abc",
                adviser_name="Jane Smith",
                zoom_id="921 2562 6617",
                client_email="test@example.com",
                call_date="27/10/2025 11:01:00",
                call_type="Starter",
                vulnerability_rating="Medium/3",
                vulnerability_type="Resilience: Low Emotional Resilience",
                evidence_quote="I feel overwhelmed"
            ),
            GroundTruthRecord(
                meeting_id="123abc",
                adviser_name="Jane Smith",
                zoom_id="921 2562 6617",
                client_email="test@example.com",
                call_date="27/10/2025 11:01:00",
                call_type="Starter",
                vulnerability_rating="Critical/5",
                vulnerability_type="Health: Chronic Illness",
                evidence_quote="I have terminal cancer"
            )
        ]

        agg = AggregatedGroundTruth.from_records(records)

        assert agg.max_vulnerability_rating == "Critical/5"
        assert agg.vulnerability_count == 2
        assert agg.expected_vulnerability_label == "vulnerable"
        assert len(agg.vulnerability_types) == 2

    def test_binary_label_threshold(self):
        """Test binary label derivation at threshold"""
        # High/4 should be vulnerable
        records_high = [
            GroundTruthRecord(
                meeting_id="123",
                adviser_name="Jane",
                zoom_id="123",
                client_email="test@example.com",
                call_date="27/10/2025",
                call_type="Starter",
                vulnerability_rating="High/4",
                vulnerability_type="Test",
                evidence_quote="Test"
            )
        ]
        agg_high = AggregatedGroundTruth.from_records(records_high)
        assert agg_high.expected_vulnerability_label == "vulnerable"

        # Medium/3 should be not_vulnerable
        records_medium = [
            GroundTruthRecord(
                meeting_id="456",
                adviser_name="Jane",
                zoom_id="123",
                client_email="test@example.com",
                call_date="27/10/2025",
                call_type="Starter",
                vulnerability_rating="Medium/3",
                vulnerability_type="Test",
                evidence_quote="Test"
            )
        ]
        agg_medium = AggregatedGroundTruth.from_records(records_medium)
        assert agg_medium.expected_vulnerability_label == "not_vulnerable"


class TestEvaluationRecord:
    """Test evaluation record creation"""

    def test_true_positive(self):
        """Test TP: predicted vulnerable, actually vulnerable"""
        gt = AggregatedGroundTruth(
            meeting_id="123",
            adviser_name="Jane",
            zoom_id="123",
            client_email="test@example.com",
            call_date="27/10/2025",
            call_type="Starter",
            max_vulnerability_rating="High/4",
            vulnerability_count=1,
            vulnerability_types=["Health: Chronic Illness"],
            evidence_quotes=["I have COPD"],
            expected_vulnerability_label="vulnerable"
        )

        pred = ModelPrediction(
            meeting_id="123",
            model_vulnerability_label="vulnerable",
            confidence=0.9,
            model_reason="Client mentioned chronic illness"
        )

        record = EvaluationRecord.from_ground_truth_and_prediction(gt, pred)

        assert record.result == "TP"
        assert record.model_vulnerability_label == "vulnerable"
        assert record.expected_vulnerability_label == "vulnerable"

    def test_true_negative(self):
        """Test TN: predicted not vulnerable, actually not vulnerable"""
        gt = AggregatedGroundTruth(
            meeting_id="123",
            adviser_name="Jane",
            zoom_id="123",
            client_email="test@example.com",
            call_date="27/10/2025",
            call_type="Starter",
            max_vulnerability_rating="Low/2",
            vulnerability_count=1,
            vulnerability_types=["Health: Temporary Illness"],
            evidence_quotes=["I had a cold"],
            expected_vulnerability_label="not_vulnerable"
        )

        pred = ModelPrediction(
            meeting_id="123",
            model_vulnerability_label="not_vulnerable",
            confidence=0.95,
            model_reason="No significant vulnerabilities"
        )

        record = EvaluationRecord.from_ground_truth_and_prediction(gt, pred)

        assert record.result == "TN"

    def test_false_positive(self):
        """Test FP: predicted vulnerable, actually not vulnerable"""
        gt = AggregatedGroundTruth(
            meeting_id="123",
            adviser_name="Jane",
            zoom_id="123",
            client_email="test@example.com",
            call_date="27/10/2025",
            call_type="Starter",
            max_vulnerability_rating="Medium/3",
            vulnerability_count=1,
            vulnerability_types=["Resilience: Low Emotional Resilience"],
            evidence_quotes=["I feel a bit stressed"],
            expected_vulnerability_label="not_vulnerable"
        )

        pred = ModelPrediction(
            meeting_id="123",
            model_vulnerability_label="vulnerable",
            confidence=0.7,
            model_reason="Client mentioned stress"
        )

        record = EvaluationRecord.from_ground_truth_and_prediction(gt, pred)

        assert record.result == "FP"

    def test_false_negative(self):
        """Test FN: predicted not vulnerable, actually vulnerable"""
        gt = AggregatedGroundTruth(
            meeting_id="123",
            adviser_name="Jane",
            zoom_id="123",
            client_email="test@example.com",
            call_date="27/10/2025",
            call_type="Starter",
            max_vulnerability_rating="Critical/5",
            vulnerability_count=1,
            vulnerability_types=["Health: Chronic Illness"],
            evidence_quotes=["I have terminal cancer"],
            expected_vulnerability_label="vulnerable"
        )

        pred = ModelPrediction(
            meeting_id="123",
            model_vulnerability_label="not_vulnerable",
            confidence=0.6,
            model_reason="No clear vulnerabilities detected"
        )

        record = EvaluationRecord.from_ground_truth_and_prediction(gt, pred)

        assert record.result == "FN"


class TestEvaluationMetrics:
    """Test metrics calculation"""

    def test_perfect_metrics(self):
        """Test metrics with perfect predictions"""
        records = [
            EvaluationRecord(
                meeting_id="1", adviser_name="Jane", zoom_id="123",
                client_email="test@example.com", call_date="27/10/2025",
                call_type="Starter", expected_vulnerability_label="vulnerable",
                max_vulnerability_rating="High/4", vulnerability_count=1,
                vulnerability_types=[], evidence_quotes=[],
                model_vulnerability_label="vulnerable", confidence=0.9,
                model_reason="", result="TP", notes_for_improvement=""
            ),
            EvaluationRecord(
                meeting_id="2", adviser_name="Jane", zoom_id="123",
                client_email="test@example.com", call_date="27/10/2025",
                call_type="Starter", expected_vulnerability_label="not_vulnerable",
                max_vulnerability_rating="Low/2", vulnerability_count=1,
                vulnerability_types=[], evidence_quotes=[],
                model_vulnerability_label="not_vulnerable", confidence=0.95,
                model_reason="", result="TN", notes_for_improvement=""
            )
        ]

        metrics = EvaluationMetrics.from_results(records, "2025-11-04 14:30:00")

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.accuracy == 1.0
        assert metrics.true_positives == 1
        assert metrics.true_negatives == 1
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0

    def test_imperfect_metrics(self):
        """Test metrics with some errors"""
        records = [
            # TP
            EvaluationRecord(
                meeting_id="1", adviser_name="Jane", zoom_id="123",
                client_email="test@example.com", call_date="27/10/2025",
                call_type="Starter", expected_vulnerability_label="vulnerable",
                max_vulnerability_rating="High/4", vulnerability_count=1,
                vulnerability_types=[], evidence_quotes=[],
                model_vulnerability_label="vulnerable", confidence=0.9,
                model_reason="", result="TP", notes_for_improvement=""
            ),
            # FN
            EvaluationRecord(
                meeting_id="2", adviser_name="Jane", zoom_id="123",
                client_email="test@example.com", call_date="27/10/2025",
                call_type="Starter", expected_vulnerability_label="vulnerable",
                max_vulnerability_rating="High/4", vulnerability_count=1,
                vulnerability_types=[], evidence_quotes=[],
                model_vulnerability_label="not_vulnerable", confidence=0.6,
                model_reason="", result="FN", notes_for_improvement=""
            ),
            # TN
            EvaluationRecord(
                meeting_id="3", adviser_name="Jane", zoom_id="123",
                client_email="test@example.com", call_date="27/10/2025",
                call_type="Starter", expected_vulnerability_label="not_vulnerable",
                max_vulnerability_rating="Low/2", vulnerability_count=1,
                vulnerability_types=[], evidence_quotes=[],
                model_vulnerability_label="not_vulnerable", confidence=0.95,
                model_reason="", result="TN", notes_for_improvement=""
            ),
            # FP
            EvaluationRecord(
                meeting_id="4", adviser_name="Jane", zoom_id="123",
                client_email="test@example.com", call_date="27/10/2025",
                call_type="Starter", expected_vulnerability_label="not_vulnerable",
                max_vulnerability_rating="Medium/3", vulnerability_count=1,
                vulnerability_types=[], evidence_quotes=[],
                model_vulnerability_label="vulnerable", confidence=0.7,
                model_reason="", result="FP", notes_for_improvement=""
            )
        ]

        metrics = EvaluationMetrics.from_results(records, "2025-11-04 14:30:00")

        # Precision = TP / (TP + FP) = 1 / (1 + 1) = 0.5
        assert metrics.precision == 0.5
        # Recall = TP / (TP + FN) = 1 / (1 + 1) = 0.5
        assert metrics.recall == 0.5
        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert metrics.f1_score == 0.5
        # Accuracy = (TP + TN) / Total = (1 + 1) / 4 = 0.5
        assert metrics.accuracy == 0.5


class TestDataLoader:
    """Test CSV data loading"""

    def test_load_example_csv(self):
        """Test loading example ground truth CSV"""
        csv_path = Path(__file__).parent.parent / "evals" / "golden_data" / "example_ground_truth.csv"

        if not csv_path.exists():
            pytest.skip("Example CSV not found")

        records = load_ground_truth_csv(csv_path)

        assert len(records) > 0
        assert all(isinstance(r, GroundTruthRecord) for r in records)

        # Check meeting IDs are cleaned
        assert all(not r.meeting_id.endswith('.mp3') for r in records)

    def test_aggregate_example_csv(self):
        """Test aggregation of example CSV"""
        csv_path = Path(__file__).parent.parent / "evals" / "golden_data" / "example_ground_truth.csv"

        if not csv_path.exists():
            pytest.skip("Example CSV not found")

        records = load_ground_truth_csv(csv_path)
        aggregated = aggregate_by_meeting(records)

        # Should have fewer aggregated records than raw records (due to multiple rows per meeting)
        assert len(aggregated) <= len(records)
        assert all(isinstance(a, AggregatedGroundTruth) for a in aggregated)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
