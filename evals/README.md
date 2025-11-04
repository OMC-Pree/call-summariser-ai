# Vulnerability Detection Evaluation Pipeline

Clean, secure, reproducible evaluation pipeline for assessing vulnerability detection in coaching calls.

## Overview

This pipeline evaluates the model's ability to detect client vulnerabilities (health issues, financial distress, life events, etc.) in coaching call transcripts.

**Key Features:**
- ✅ Clean & normalize ground truth data from Google Sheets
- ✅ Aggregate multiple vulnerability instances per meeting
- ✅ Load predictions from S3 (case checks) or CSV
- ✅ Compute Precision, Recall, F1, Confusion Matrix
- ✅ Versioned, append-only output (never mutates golden dataset)
- ✅ Detailed error analysis (FP/FN with context)

## Quick Start

### 1. Prepare Ground Truth CSV

Export your Google Sheet with manually filled vulnerability labels to CSV:

```bash
# Place in evals/golden_data/
cp ~/Downloads/vulnerability_labels.csv evals/golden_data/vulnerability_ground_truth.csv
```

**Required columns:**
- `meeting_id` - Meeting ID (with or without .mp3 extension)
- `adviser_name` - Coach name
- `zoom_id` - Zoom meeting ID
- `client_email` - Client email
- `call_date` - Call date
- `call_type` - Session type (e.g., "Starter")
- `vulnerability_rating` - One of: Critical/5, High/4, Medium/3, Low/2, None/1
- `vulnerability_type` - Category (e.g., "Health: Chronic Illness")
- `evidence_quote` - Supporting quote from transcript

### 2. Run Evaluation

**Option A: Load predictions from S3 (case checks)**

```bash
python evals/run_evaluation.py \
    --ground-truth evals/golden_data/vulnerability_ground_truth.csv \
    --source s3 \
    --year 2025 \
    --month 10
```

**Option B: Load predictions from CSV**

```bash
python evals/run_evaluation.py \
    --ground-truth evals/golden_data/vulnerability_ground_truth.csv \
    --source csv \
    --predictions evals/golden_data/model_predictions.csv
```

### 3. Review Results

Results are saved to `evals/runs/{timestamp}/`:

```
evals/runs/20251104_143022/
├── evaluation_results.csv          # Per-meeting results (TP/TN/FP/FN)
├── metrics_summary.json             # Precision/Recall/F1/Confusion Matrix
├── aggregated_ground_truth.csv      # Cleaned ground truth (one row per meeting)
└── run_config.json                  # Reproducibility metadata
```

## Architecture

### Data Flow

```
┌─────────────────────┐
│  Google Sheets      │
│  (Manual Labels)    │
└──────────┬──────────┘
           │ export CSV
           ▼
┌─────────────────────┐     ┌──────────────────┐
│  Ground Truth CSV   │     │  S3 Case Checks  │
│  (multiple rows     │     │  or Predictions  │
│   per meeting)      │     │  CSV             │
└──────────┬──────────┘     └────────┬─────────┘
           │                         │
           │ clean &                 │ load
           │ aggregate               │ predictions
           ▼                         ▼
┌─────────────────────┐     ┌──────────────────┐
│  Aggregated GT      │────▶│  Match & Score   │
│  (1 row/meeting)    │     │                  │
└─────────────────────┘     └────────┬─────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │  Evaluation      │
                            │  Results + Metrics│
                            │  (versioned run) │
                            └──────────────────┘
```

### Binary Classification

Vulnerability ratings are mapped to binary labels:

| Rating      | Binary Label      | Description                          |
|-------------|-------------------|--------------------------------------|
| Critical/5  | `vulnerable`      | Severe vulnerability requiring action|
| High/4      | `vulnerable`      | Significant vulnerability            |
| Medium/3    | `not_vulnerable`  | Minor concern, adequately handled    |
| Low/2       | `not_vulnerable`  | Very minor or contextual mention     |
| None/1      | `not_vulnerable`  | No vulnerability detected            |

**Rationale:** High/4 and Critical/5 require immediate attention and special handling, so they're classified as "vulnerable" for evaluation purposes.

### Aggregation Logic

Meetings may have multiple vulnerability instances (e.g., one call mentions both chronic illness and bereavement). The pipeline:

1. Groups all rows by `meeting_id`
2. Takes **maximum vulnerability rating** across all instances
3. Collects all vulnerability types and evidence quotes
4. Derives binary label from max rating

**Example:**

```csv
meeting_id,vulnerability_rating,vulnerability_type,evidence_quote
123.mp3,High/4,Health: Chronic Illness,"I have COPD..."
123.mp3,High/4,Life Events: Bereavement,"My mom passed away..."
```

Aggregates to:

```python
{
  "meeting_id": "123",
  "max_vulnerability_rating": "High/4",
  "vulnerability_count": 2,
  "vulnerability_types": ["Health: Chronic Illness", "Life Events: Bereavement"],
  "expected_vulnerability_label": "vulnerable"  # High/4 >= 4
}
```

## Prediction Sources

### S3 Case Checks (Recommended)

Loads predictions from S3 case check results produced by the case_check Lambda:

```python
# S3 path format:
s3://{SUMMARY_BUCKET}/summaries/supplementary/version={VERSION}/
  year={YEAR}/month={MONTH}/meeting_id={MEETING_ID}/
  case_check.v1.0.json
```

The pipeline:
1. Finds the `vulnerability_identified` check in results
2. Maps status to binary label:
   - `Fail` → `vulnerable` (vulnerability NOT handled appropriately)
   - `Competent` → `not_vulnerable` (no vulnerability or handled well)
   - `CompetentWithDevelopment` → `not_vulnerable` (addressed but improvable)
   - Other statuses → `not_vulnerable` (conservative mapping)

### CSV Predictions (For Testing)

Create a CSV with columns:

```csv
meeting_id,model_vulnerability_label,confidence,model_reason
123,vulnerable,0.85,"Client mentioned chronic illness"
456,not_vulnerable,0.92,"No vulnerabilities detected"
```

## Metrics

### Confusion Matrix

|                | Predicted Vulnerable | Predicted Not Vulnerable |
|----------------|---------------------|-------------------------|
| **Actually Vulnerable** | TP (True Positive)  | FN (False Negative)     |
| **Actually Not Vulnerable** | FP (False Positive) | TN (True Negative)      |

### Performance Metrics

- **Precision**: TP / (TP + FP) - Of all predicted vulnerabilities, how many were correct?
- **Recall**: TP / (TP + FN) - Of all actual vulnerabilities, how many did we detect?
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall) - Harmonic mean
- **Accuracy**: (TP + TN) / Total - Overall correctness

### Example Output

```
==============================================================
EVALUATION METRICS - 2025-11-04 14:30:22
==============================================================

Total Samples: 50
  - Vulnerable: 30
  - Not Vulnerable: 20

Confusion Matrix:
  True Positives  (TP):  27
  True Negatives  (TN):  18
  False Positives (FP):   2
  False Negatives (FN):   3

Performance Metrics:
  Precision: 0.9310 (93.10%)
  Recall:    0.9000 (90.00%)
  F1 Score:  0.9153 (91.53%)
  Accuracy:  0.9000 (90.00%)
==============================================================
```

## Error Analysis

The pipeline provides detailed error analysis showing:

### False Positives (FP)
Model predicted vulnerable, but actually not vulnerable.

**Common causes:**
- Vulnerability mentioned but already well-handled by coach
- Client mentioned someone else's vulnerability (not their own)
- Historical vulnerability that's resolved

### False Negatives (FN)
Model missed actual vulnerabilities.

**Common causes:**
- Subtle language (e.g., "I'm a bit stressed" when it's severe anxiety)
- Vulnerability buried in long conversation
- Implicit vulnerability (e.g., single parent + low income = caring responsibilities)

Example output:

```
==============================================================
FALSE NEGATIVES (3 total) - Model missed vulnerabilities
==============================================================

1. Meeting: 79fa6592-c0d2-4ae4-b7f5-db2df01f3025
   Adviser: Laura Zaccagnini
   Ground Truth: vulnerable (High/4)
   Vulnerability Types: Health: Chronic Illness, Life Events: Caring
   Model Predicted: not_vulnerable (confidence: 0.75)
   Reason: Status: Competent. No significant vulnerabilities detected
   Evidence: "I have Crohn's disease and I'm caring for my elderly mother..."
```

## Security & Best Practices

### Never Mutate Golden Dataset

The golden dataset (ground truth CSV) is **append-only**. The pipeline:
- ✅ Reads from `evals/golden_data/`
- ✅ Writes to `evals/runs/{timestamp}/` (new directory each run)
- ❌ Never modifies files in `evals/golden_data/`

### Versioning

Each run creates a new timestamped directory:

```bash
evals/runs/
├── 20251104_143022/    # First run
├── 20251104_151530/    # Second run (different data or config)
└── 20251105_090000/    # Third run
```

This enables:
- Compare performance across model versions
- Track improvements over time
- Reproduce historical results

### Data Privacy

- Ground truth CSV should contain only meeting_id, not full transcripts
- Evidence quotes are manually selected excerpts (PII already reviewed)
- Never commit actual CSV files with client emails to git
- Use `.gitignore` to exclude `evals/golden_data/*.csv` and `evals/runs/`

## Development

### Run Tests

```bash
pytest tests/test_evals.py -v
```

### Add New Vulnerability Categories

1. Update Google Sheet with new categories
2. Re-export to CSV
3. Run evaluation pipeline (no code changes needed)

### Customize Binary Label Threshold

Edit [evals/models.py](evals/models.py):

```python
# Current: High/4 and Critical/5 = vulnerable
# To change threshold to Medium/3+:
expected_label: BinaryLabel = (
    "vulnerable" if rating_order[max_rating.vulnerability_rating] >= 3  # Changed from 4
    else "not_vulnerable"
)
```

## Troubleshooting

### "No predictions matched ground truth"

**Cause:** Meeting IDs don't align between ground truth and predictions.

**Fix:**
- Ensure meeting_ids in ground truth CSV match those in S3 or predictions CSV
- Remove `.mp3` extensions (pipeline auto-cleans but check consistency)
- Check year/month parameters for S3 source

### "CSV missing required columns"

**Cause:** Ground truth CSV doesn't have all required columns.

**Fix:**
```bash
# Check CSV headers
head -n 1 evals/golden_data/vulnerability_ground_truth.csv
```

Required: `meeting_id,adviser_name,zoom_id,client_email,call_date,call_type,vulnerability_rating,vulnerability_type,evidence_quote`

### "Case check for {meeting_id} missing 'vulnerability_identified' check"

**Cause:** Case check JSON doesn't have the expected check ID.

**Fix:**
- Verify case check Lambda is using latest checklist with `vulnerability_identified`
- Re-run case checks for missing meetings
- Use CSV predictions instead of S3 as workaround

## Roadmap

Future enhancements:

- [ ] Multi-class classification (predict specific vulnerability types)
- [ ] Confidence calibration analysis (are 0.9 predictions actually 90% accurate?)
- [ ] Temporal analysis (performance over time)
- [ ] Coach-level metrics (which coaches have most missed vulnerabilities)
- [ ] Integration with A2I for human review of FP/FN cases

## License

Internal use only. Do not distribute.
