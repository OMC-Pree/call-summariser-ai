"""
Evaluation pipeline for vulnerability detection in coaching calls.

This module provides tools for:
- Cleaning and normalizing ground truth data from Google Sheets
- Aggregating multiple vulnerability instances per meeting
- Loading model predictions from S3 or CSV
- Computing evaluation metrics (Precision, Recall, F1, Confusion Matrix)
- Versioned output with append-only runs
"""
