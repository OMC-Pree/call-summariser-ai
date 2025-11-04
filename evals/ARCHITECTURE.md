# Evaluation Pipeline Architecture

## Directory Structure

```
evals/
├── __init__.py                     # Package initialization
├── models.py                       # Pydantic data models
├── data_loader.py                  # CSV loading & aggregation
├── prediction_loader.py            # S3 & CSV prediction loaders
├── scorer.py                       # Metrics & evaluation logic
├── run_evaluation.py               # Main CLI runner
├── README.md                       # Full documentation
├── QUICKSTART.md                   # 5-minute getting started guide
├── ARCHITECTURE.md                 # This file
├── .gitignore                      # Excludes PII data
├── golden_data/                    # Ground truth datasets
│   ├── README.md                   # Data preparation guide
│   ├── example_ground_truth.csv    # Template (safe to commit)
│   └── vulnerability_ground_truth.csv  # Real data (gitignored)
└── runs/                           # Versioned evaluation runs (gitignored)
    └── {timestamp}/                # Each run in separate directory
        ├── evaluation_results.csv  # Per-meeting results (TP/TN/FP/FN)
        ├── metrics_summary.json    # Aggregated metrics
        ├── aggregated_ground_truth.csv  # Cleaned ground truth
        └── run_config.json         # Reproducibility metadata
```

## Module Responsibilities

### `models.py` - Data Models

**Purpose:** Define all data structures with Pydantic validation.

**Key models:**
- `GroundTruthRecord` - Single row from CSV (may be multiple per meeting)
- `AggregatedGroundTruth` - One row per meeting with max vulnerability rating
- `ModelPrediction` - Model's vulnerability prediction
- `EvaluationRecord` - Combined ground truth + prediction with result (TP/TN/FP/FN)
- `EvaluationMetrics` - Summary metrics (Precision/Recall/F1/confusion matrix)

**Example:**
```python
from evals.models import GroundTruthRecord, AggregatedGroundTruth

# Load single record
record = GroundTruthRecord(
    meeting_id="935 9850 2685",  # Auto-cleaned to "93598502685"
    vulnerability_rating="High/4",
    ...
)

# Aggregate multiple records for one meeting
agg = AggregatedGroundTruth.from_records([record1, record2])
# agg.expected_vulnerability_label = "vulnerable"  (High/4 >= 4)
```

### `data_loader.py` - Ground Truth Processing

**Purpose:** Load and clean ground truth data from Google Sheets CSV.

**Key functions:**
- `load_ground_truth_csv(csv_path)` - Load & validate CSV
- `aggregate_by_meeting(records)` - Group by meeting_id, take max rating
- `save_aggregated_csv(aggregated, output_path)` - Save cleaned data

**Data cleaning:**
- Strips whitespace from all fields
- meeting_id is the zoom_id (numeric identifier)
- Removes spaces and hyphens from meeting_id (e.g., "935 9850 2685" → "93598502685")
- Normalizes column names (case-insensitive)
- Skips empty rows and invalid meeting_ids with warnings
- Validates with Pydantic
- Detects duplicate predictions before evaluation

**Example:**
```python
from evals.data_loader import load_ground_truth_csv, aggregate_by_meeting

# Load CSV (may have multiple rows per meeting)
records = load_ground_truth_csv("evals/golden_data/vulnerability_ground_truth.csv")
# Returns: [GroundTruthRecord, GroundTruthRecord, ...]

# Aggregate to one row per meeting
aggregated = aggregate_by_meeting(records)
# Returns: [AggregatedGroundTruth, AggregatedGroundTruth, ...]
```

### `prediction_loader.py` - Model Predictions

**Purpose:** Load model predictions from S3 or CSV.

**Key functions:**
- `load_predictions_from_s3(meeting_ids, year, month)` - Fetch from S3 case checks
- `load_predictions_from_csv(csv_path)` - Load from CSV file
- `extract_vulnerability_prediction(case_check_data)` - Parse case check JSON

**S3 prediction logic:**
```python
# Finds "vulnerability_identified" check in case check JSON
# Maps status to binary label:
#   "Fail" → vulnerable (NOT handled appropriately)
#   "Competent" → not_vulnerable (no vuln or handled well)
#   "CompetentWithDevelopment" → not_vulnerable (addressed)
#   "NotApplicable" → not_vulnerable
#   "Inconclusive" → not_vulnerable (conservative)
```

**Example:**
```python
from evals.prediction_loader import load_predictions_from_s3

# Load from S3 case checks
predictions = load_predictions_from_s3(
    meeting_ids=["123", "456", "789"],
    year=2025,
    month=10
)
# Returns: [ModelPrediction, ModelPrediction, ...]
```

### `scorer.py` - Evaluation & Metrics

**Purpose:** Match predictions to ground truth, compute metrics, analyze errors.

**Key functions:**
- `match_predictions_to_ground_truth(gt, predictions)` - Join by meeting_id
- `compute_metrics(evaluation_records, timestamp)` - Calculate P/R/F1
- `print_metrics_summary(metrics)` - Console output
- `print_error_analysis(records, show_limit)` - Show FP/FN examples

**Confusion matrix:**
```
                    Predicted Vulnerable | Predicted Not Vulnerable
Actually Vulnerable       TP              |        FN
Actually Not Vulnerable   FP              |        TN
```

**Metrics:**
- **Precision** = TP / (TP + FP) - "Of predicted vulnerabilities, how many were correct?"
- **Recall** = TP / (TP + FN) - "Of actual vulnerabilities, how many did we catch?"
- **F1** = 2 × (P × R) / (P + R) - Harmonic mean of precision and recall
- **Accuracy** = (TP + TN) / Total - Overall correctness

**Example:**
```python
from evals.scorer import match_predictions_to_ground_truth, compute_metrics

# Match and evaluate
eval_records, missing = match_predictions_to_ground_truth(
    ground_truth=aggregated_gt,
    predictions=predictions
)

# Compute metrics
metrics = compute_metrics(eval_records, timestamp="2025-11-04 14:30:00")
print(f"F1 Score: {metrics.f1_score:.4f}")
```

### `run_evaluation.py` - Main Pipeline

**Purpose:** Orchestrate full evaluation workflow from CLI.

**Pipeline steps:**
1. Load ground truth CSV → `List[GroundTruthRecord]`
2. Aggregate by meeting → `List[AggregatedGroundTruth]`
3. Load predictions (S3 or CSV) → `List[ModelPrediction]`
4. Match & evaluate → `List[EvaluationRecord]`
5. Compute metrics → `EvaluationMetrics`
6. Save results → `evals/runs/{timestamp}/`
7. Display summary & errors

**CLI usage:**
```bash
# From S3
python evals/run_evaluation.py \
    --ground-truth evals/golden_data/vulnerability_ground_truth.csv \
    --source s3 \
    --year 2025 \
    --month 10

# From CSV
python evals/run_evaluation.py \
    --ground-truth evals/golden_data/vulnerability_ground_truth.csv \
    --source csv \
    --predictions evals/golden_data/predictions.csv \
    --show-errors 10
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     GROUND TRUTH PREPARATION                     │
│                                                                  │
│  1. Manual Review → Google Sheets → Export CSV                  │
│     - Reviewers label vulnerabilities (Critical/5 to None/1)    │
│     - May create multiple rows per meeting (multiple vulns)     │
│                                                                  │
│  2. Save to: evals/golden_data/vulnerability_ground_truth.csv   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LOADING & CLEANING                     │
│                      (data_loader.py)                            │
│                                                                  │
│  load_ground_truth_csv()                                        │
│    ├─ Normalize column names                                    │
│    ├─ Clean meeting_id (remove spaces/hyphens from zoom_id)     │
│    ├─ Skip empty rows and invalid meeting_ids                   │
│    └─ Validate with Pydantic → List[GroundTruthRecord]          │
│                                                                  │
│  aggregate_by_meeting()                                         │
│    ├─ Group by meeting_id                                       │
│    ├─ Take max vulnerability rating                             │
│    ├─ Collect all types & evidence                              │
│    └─ Derive binary label → List[AggregatedGroundTruth]         │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION LOADING                            │
│                  (prediction_loader.py)                          │
│                                                                  │
│  Option A: S3 Case Checks                                       │
│    load_predictions_from_s3()                                   │
│      ├─ For each meeting_id                                     │
│      ├─ Fetch case check JSON from S3                           │
│      ├─ Extract "vulnerability_identified" check                │
│      ├─ Map status to binary label                              │
│      └─ Return List[ModelPrediction]                            │
│                                                                  │
│  Option B: CSV File                                             │
│    load_predictions_from_csv()                                  │
│      ├─ Load CSV with predictions                               │
│      ├─ Validate required columns                               │
│      └─ Return List[ModelPrediction]                            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MATCHING & SCORING                            │
│                       (scorer.py)                                │
│                                                                  │
│  match_predictions_to_ground_truth()                            │
│    ├─ Join ground truth & predictions by meeting_id             │
│    ├─ For each match:                                           │
│    │   ├─ Compare expected vs predicted labels                  │
│    │   └─ Assign result (TP/TN/FP/FN)                           │
│    └─ Return List[EvaluationRecord]                             │
│                                                                  │
│  compute_metrics()                                              │
│    ├─ Count TP/TN/FP/FN                                         │
│    ├─ Calculate Precision/Recall/F1/Accuracy                    │
│    └─ Return EvaluationMetrics                                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT & REPORTING                            │
│                  (run_evaluation.py)                             │
│                                                                  │
│  Create versioned output directory:                             │
│    evals/runs/{timestamp}/                                      │
│                                                                  │
│  Save files:                                                    │
│    ├─ evaluation_results.csv      (per-meeting TP/TN/FP/FN)    │
│    ├─ metrics_summary.json        (P/R/F1/confusion matrix)    │
│    ├─ aggregated_ground_truth.csv (cleaned ground truth)       │
│    └─ run_config.json             (reproducibility metadata)   │
│                                                                  │
│  Console output:                                                │
│    ├─ Metrics summary table                                     │
│    └─ Error analysis (FP/FN examples)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Binary Classification Threshold

**Question:** Why High/4 and Critical/5 = vulnerable, but Medium/3 = not vulnerable?

**Answer:** Regulatory and operational requirements.

| Rating      | Severity | Requires Action? | Binary Label      |
|-------------|----------|------------------|-------------------|
| Critical/5  | Severe   | Yes (urgent)     | `vulnerable`      |
| High/4      | Significant | Yes (soon)    | `vulnerable`      |
| Medium/3    | Minor    | No (monitor)     | `not_vulnerable`  |
| Low/2       | Very minor | No             | `not_vulnerable`  |
| None/1      | None     | No               | `not_vulnerable`  |

**High/4 examples:**
- Chronic illness (COPD, diabetes requiring management)
- Recent bereavement (grieving parent)
- Financial distress (debt, redundancy)
- Caring responsibilities (elderly parent, disabled child)

**Medium/3 examples:**
- Temporary stress ("I'm feeling a bit overwhelmed")
- Minor health issues (cold, minor surgery recovery)
- Low emotional resilience (mentioned but not severe)

**Threshold rationale:**
- High/4+ requires coach to take specific actions (escalate, adjust advice)
- Medium/3 requires awareness but no special handling
- Binary classification simplifies evaluation (detect cases needing action)

## Security & Privacy

### Never Mutate Golden Dataset

The golden dataset is **append-only**:

```python
# ✅ GOOD: Read from golden_data, write to runs/
load_ground_truth_csv("evals/golden_data/vulnerability_ground_truth.csv")
save_results(f"evals/runs/{timestamp}/evaluation_results.csv")

# ❌ BAD: Never write back to golden_data
# save_results("evals/golden_data/vulnerability_ground_truth.csv")  # NO!
```

### Versioning Strategy

Each evaluation run creates a new timestamped directory:

```bash
evals/runs/
├── 20251104_143022/  # Run 1 (baseline model)
├── 20251105_090000/  # Run 2 (improved prompt)
└── 20251106_153045/  # Run 3 (Claude 3.7)
```

Benefits:
- Compare performance across model versions
- Track improvements over time
- Reproduce historical results
- Never lose data (append-only)

### PII Handling

- Ground truth CSV contains client emails and meeting IDs (PII)
- `.gitignore` excludes `evals/golden_data/*.csv` and `evals/runs/`
- Only commit example/template files
- Store backups in secure S3 bucket (not public repos)

## Extending the Pipeline

### Add New Vulnerability Categories

No code changes needed! Just update Google Sheet:

1. Add new vulnerability_type in Google Sheet
2. Export to CSV
3. Run evaluation pipeline

The pipeline dynamically handles all categories.

### Change Binary Threshold

Edit `evals/models.py`, `AggregatedGroundTruth.from_records()`:

```python
# Current: High/4 and Critical/5 = vulnerable
expected_label: BinaryLabel = (
    "vulnerable" if rating_order[max_rating.vulnerability_rating] >= 4
    else "not_vulnerable"
)

# Change to: Medium/3+ = vulnerable
expected_label: BinaryLabel = (
    "vulnerable" if rating_order[max_rating.vulnerability_rating] >= 3  # Changed
    else "not_vulnerable"
)
```

### Multi-Class Classification

Current pipeline is binary (vulnerable / not vulnerable). To evaluate specific types:

1. Update `ModelPrediction` to include `predicted_type: str`
2. Add per-class metrics in `EvaluationMetrics`
3. Compute precision/recall for each vulnerability type
4. Update `scorer.py` to handle multi-class confusion matrix

## Performance Considerations

### S3 Prediction Loading

- Fetches case checks in parallel (could parallelize with ThreadPoolExecutor)
- Missing case checks are warned but don't fail the pipeline
- Typical latency: ~50-200ms per S3 fetch (depends on network)

### CSV Processing

- Uses standard Python CSV reader (efficient for 100s-1000s of rows)
- For very large datasets (100K+ rows), consider pandas or chunking

### Memory Usage

- All data held in memory (fine for 1000s of meetings)
- For 100K+ meetings, implement streaming or chunked processing

## Error Handling

### Graceful Degradation

- Missing predictions → warn and continue with available data
- Malformed CSV rows → log error with row number and context
- S3 fetch failures → retry with exponential backoff (handled by AWS SDK)

### Validation

- Pydantic validates all data at boundaries
- Invalid vulnerability ratings → ValidationError
- Missing required columns → ValueError with helpful message

## Testing

See `tests/test_evals.py` for comprehensive tests:

```bash
pytest tests/test_evals.py -v
```

Tests cover:
- Ground truth record cleaning
- Aggregation logic (single & multiple vulnerabilities)
- Binary label threshold (High/4 vs Medium/3)
- Confusion matrix (TP/TN/FP/FN)
- Metrics calculation (perfect & imperfect scores)
- CSV loading (example dataset)

## Future Enhancements

- [ ] **Confidence calibration:** Are 0.9 predictions actually 90% accurate?
- [ ] **Temporal analysis:** Performance trends over time
- [ ] **Coach-level metrics:** Which coaches have most missed vulnerabilities?
- [ ] **A2I integration:** Send FP/FN cases to human review
- [ ] **Multi-class evaluation:** Predict specific vulnerability types
- [ ] **Explainability:** SHAP/LIME for model decisions

## References

- **Ground Truth CSV:** `evals/golden_data/vulnerability_ground_truth.csv`
- **Case Check Lambda:** `summariser/case_check/app.py`
- **Vulnerability Taxonomy:** See Google Sheet (FCA Consumer Vulnerability Guidance)
