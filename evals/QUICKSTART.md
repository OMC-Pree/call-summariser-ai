# Vulnerability Detection Evaluation - Quick Start

Get started with the evaluation pipeline in 5 minutes.

## Prerequisites

```bash
# Ensure you're in the project root
cd /Users/pree/Documents/AI\ POCs/call-summariser/call-summariser

# Install dependencies (if needed)
pip install pydantic boto3
```

## Step 1: Prepare Your Ground Truth CSV

Export your Google Sheet to CSV and place it in `evals/golden_data/`:

```bash
# Example: Copy from Downloads
cp ~/Downloads/vulnerability_labels.csv evals/golden_data/vulnerability_ground_truth.csv
```

**Required CSV columns:**

```
meeting_id, adviser_name, zoom_id, client_email, call_date, call_type,
vulnerability_rating, vulnerability_type, evidence_quote
```

**Vulnerability ratings:**

- `Critical/5` - Severe (e.g., terminal illness, domestic violence)
- `High/4` - Significant (e.g., chronic illness, bereavement)
- `Medium/3` - Minor concern (e.g., temporary stress)
- `Low/2` - Very minor (e.g., passing mention)
- `Marginal/1` - Very very minor
- `None/0` - No vulnerability

**Binary classification:** High/4 and Critical/5 → `vulnerable`, others → `not_vulnerable`

## Step 2: Run Evaluation

### Option A: From S3 (Recommended)

Load predictions from S3 case check results:

```bash
python evals/run_evaluation.py \
    --ground-truth evals/golden_data/vulnerability_ground_truth.csv \
    --source s3 \
    --year 2025 \
    --month 10
```

### Option B: From CSV (For Testing)

Create a predictions CSV with columns: `meeting_id,model_vulnerability_label,confidence,model_reason`

```bash
python evals/run_evaluation.py \
    --ground-truth evals/golden_data/vulnerability_ground_truth.csv \
    --source csv \
    --predictions evals/golden_data/model_predictions.csv
```

## Step 3: Review Results

Results are saved to `evals/runs/{timestamp}/`:

```bash
# View metrics summary
cat evals/runs/20251104_143022/metrics_summary.json

# Open results CSV
open evals/runs/20251104_143022/evaluation_results.csv
```

**Console output shows:**

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

## Common Issues

### "No valid records found in CSV"

**Fix:** Check CSV has data rows (not just headers) and required columns.

```bash
# Check CSV structure
head evals/golden_data/vulnerability_ground_truth.csv
```

### "No predictions matched ground truth"

**Fix:** Ensure meeting_ids match between ground truth and predictions.

```bash
# Check meeting IDs in ground truth
cut -d',' -f1 evals/golden_data/vulnerability_ground_truth.csv | sort | uniq
```

### "CSV missing required columns"

**Fix:** Verify all required columns are present (see template):

```bash
# Use example CSV as reference
cat evals/golden_data/example_ground_truth.csv
```

## Next Steps

- **Analyze errors:** Check `evaluation_results.csv` for FP/FN cases
- **Improve model:** Use FN examples to enhance prompts
- **Track progress:** Compare runs over time
- **Automate:** Set up cron job for regular evaluation

## Example: Full Workflow

```bash
# 1. Export Google Sheet to CSV
# (Manual step in Google Sheets UI)

# 2. Move to golden_data/
mv ~/Downloads/vulnerability_labels.csv evals/golden_data/vulnerability_ground_truth.csv

# 3. Run evaluation
python evals/run_evaluation.py \
    --ground-truth evals/golden_data/vulnerability_ground_truth.csv \
    --source s3 \
    --year 2025 \
    --month 10 \
    --show-errors 10

# 4. Review results
ls -lh evals/runs/$(ls -t evals/runs | head -1)/

# 5. Analyze false negatives
grep "FN" evals/runs/$(ls -t evals/runs | head -1)/evaluation_results.csv
```

## Testing

Run tests to verify installation:

```bash
pytest tests/test_evals.py -v
```

Expected output: `13 passed`

## Help

```bash
# Get command-line help
python evals/run_evaluation.py --help

# Read full documentation
cat evals/README.md
```
