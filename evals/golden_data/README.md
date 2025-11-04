# Golden Data Directory

This directory contains ground truth data for vulnerability detection evaluation.

## Files

- **`vulnerability_ground_truth.csv`** - Main ground truth dataset (exported from Google Sheets)
  - Never committed to git (contains PII)
  - Manually filled by human reviewers
  - One or more rows per meeting (multiple vulnerabilities possible)

- **`example_ground_truth.csv`** - Template showing required format
  - Safe to commit (synthetic data)
  - Use as reference when creating new ground truth files

## Creating Ground Truth

1. **Review coaching calls** and manually identify vulnerabilities
2. **Fill Google Sheet** with required columns:
   - meeting_id
   - adviser_name
   - zoom_id
   - client_email
   - call_date
   - call_type
   - vulnerability_rating (Critical/5, High/4, Medium/3, Low/2, None/1)
   - vulnerability_type (e.g., "Health: Chronic Illness")
   - evidence_quote (supporting quote from transcript)

3. **Export to CSV** and place here as `vulnerability_ground_truth.csv`

4. **Run evaluation pipeline**:
   ```bash
   python evals/run_evaluation.py \
       --ground-truth evals/golden_data/vulnerability_ground_truth.csv \
       --source s3 \
       --year 2025 \
       --month 10
   ```

## Security

- **Never commit** actual ground truth CSV files (contains client PII)
- **Always use** `.gitignore` to exclude `*.csv` (except examples)
- **Store backups** in secure location (not public repos)
