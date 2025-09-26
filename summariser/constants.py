# summariser/constants.py
"""
Centralized constants for the Call Summarizer application.
These values are used across multiple modules to maintain consistency.
"""

import os

# AWS Configuration
DEFAULT_REGION = "eu-west-2"
AWS_REGION = os.getenv("AWS_REGION", DEFAULT_REGION)

# Model Configuration
DEFAULT_MODEL_VERSION = "bedrock:claude-3-sonnet-20240229"
DEFAULT_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
DEFAULT_PROMPT_VERSION = "2025-09-22-a"
DEFAULT_INSIGHTS_VERSION = "2025-08-30-a"
DEFAULT_SCHEMA_VERSION = "1.2"
DEFAULT_CASE_CHECK_SCHEMA_VERSION = "1.0"

# S3 Configuration
DEFAULT_S3_PREFIX = "summaries"

# SSM Configuration
DEFAULT_ZOOM_PARAM_PREFIX = "/zoom/s2s"

# Required Environment Variables (validated at runtime)
def get_required_env(key: str) -> str:
    """Get required environment variable, fail gracefully during development"""
    value = os.environ.get(key)
    if not value:
        # In production Lambda, these will be set
        # During development/testing, provide helpful error
        if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            raise ValueError(f"Required environment variable {key} not set")
        return f"MISSING_{key}"
    return value

SUMMARY_JOB_TABLE = get_required_env("SUMMARY_JOB_TABLE")
SUMMARY_BUCKET = get_required_env("SUMMARY_BUCKET")

# Environment-driven configuration with defaults
MODEL_VERSION = os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION)
MODEL_ID = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
PROMPT_VERSION = os.getenv("PROMPT_VERSION", DEFAULT_PROMPT_VERSION)
INSIGHTS_VERSION = os.getenv("INSIGHTS_VERSION", DEFAULT_INSIGHTS_VERSION)
SCHEMA_VERSION = os.getenv("SUMMARY_SCHEMA_VERSION", DEFAULT_SCHEMA_VERSION)
CASE_CHECK_SCHEMA_VERSION = os.getenv("CASE_CHECK_SCHEMA_VERSION", DEFAULT_CASE_CHECK_SCHEMA_VERSION)
S3_PREFIX = os.getenv("S3_PREFIX", DEFAULT_S3_PREFIX)
ZOOM_PARAM_PREFIX = os.getenv("ZOOM_PARAM_PREFIX", DEFAULT_ZOOM_PARAM_PREFIX)

# Feature Flags
SAVE_TRANSCRIPTS = os.getenv("SAVE_TRANSCRIPTS", "false").lower() == "true"
ATHENA_PARTITIONED = os.getenv("ATHENA_PARTITIONED", "true").lower() == "true"

# A2I Configuration
A2I_FLOW_ARN_CASE = os.getenv("A2I_FLOW_ARN_CASE")
A2I_PORTAL_URL = os.getenv("A2I_PORTAL_URL", "")