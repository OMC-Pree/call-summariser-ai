# summariser/prompts.py
"""
Centralized prompt templates for the Call Summarizer application.
These prompts are used for LLM interactions across the application.
"""

# Prompt Version Constants
SUMMARY_PROMPT_VERSION = "2025-09-25-a"
CASE_CHECK_PROMPT_VERSION = "2025-09-25-a"
CASE_CHECK_SCHEMA_VERSION = "1.1"
JSON_REPAIR_PROMPT_VERSION = "2025-09-25-a"

# Summary Generation Prompt
SUMMARY_PROMPT_TEMPLATE = """You are a financial assistant summarizing a customer coaching call.
Return ONLY a single JSON object. Do not add any explanation, preface, or markdown fences.
The JSON must follow this format exactly:
{
  "summary": string,
  "key_points": [string],
  "action_items": [ { "description": string } ],
  "sentiment_analysis": {
    "label": "Positive" | "Neutral" | "Negative",
    "confidence": number (0-1)
  },
  "themes": [
     { "id": string, "label": string, "group": string, "confidence": number (0-1), "evidence_quote": string | null }
  ]
}
Rules:
- Use British English and GBP where relevant.
- Pick 0–7 themes from a controlled list (Budgeting, ISA, Mortgage, Pension, Protection, Debt, etc.).
- Only include a theme if clearly supported by transcript.
- Provide short evidence quotes when possible.

Transcript:
{transcript}"""

# Case Check Prompt Template
CASE_CHECK_PROMPT_TEMPLATE = """You are auditing a financial coaching call against a checklist.
Return STRICT JSON only. Do NOT include any preface, headings, or code fences.
The FIRST character of your response must be '{{' and the LAST must be '}}'.
Schema:
{{
  "check_schema_version": "1.0",
  "session_type": "starter_session",
  "checklist_version": "1",
  "meeting_id": string,
  "model_version": string,
  "prompt_version": string,
  "results": [
    {{"id": string, "status": "Pass" | "Fail" | "NotApplicable" | "Inconclusive", "confidence": number (0-1), "evidence_spans": [[start,end]], "evidence_quote": string, "comment": string}}
  ],
  "overall": {{"pass_rate": number, "failed_ids": [string], "high_severity_flags": [string]}}
}}
RULES:
- Use 'Pass' only when the coach clearly fulfilled the requirement.
- Use 'Fail' when the requirement was missed, incorrect, or explicitly prohibited.
- Use 'NotApplicable' if the check does not apply to this session.
- Use 'Inconclusive' if there's not enough evidence to make a judgment.
- Set 'evidence_quote' and 'comment' to empty string ('') if unavailable.
- If the coach gives regulated financial advice (e.g., specific product or investment recommendations), this is NOT permitted — set `regulated_advice_given` to 'Fail'.

- Set meeting_id EXACTLY to "{meeting_id}".
- Set model_version EXACTLY to "{model_version}".
- Set prompt_version EXACTLY to "{prompt_version}".
- "evidence_quote" and "comment" must be strings; if unavailable use "" (empty string), not null.
- If any other value is unknown, use null (not placeholders like "[REDACTED]").

CHECKLIST:
{checklist_json}

CLEANED_TRANSCRIPT:
{cleaned_transcript}"""

# JSON Repair Prompt Template
JSON_REPAIR_PROMPT_TEMPLATE = """The following is intended to be a JSON object matching the case check schema, but it may be malformed or truncated. Return ONLY a single, valid JSON object that starts with '{{' and ends with '}}'. Do not include any preface, code fences, or comments.

----- BEGIN -----
{bad_json}
----- END -----"""

# System Messages
SUMMARY_SYSTEM_MESSAGE = "Output ONLY one JSON object that starts with '{' and ends with '}'. No preface. No code fences."

CASE_CHECK_SYSTEM_MESSAGE = "Output ONLY one JSON object that starts with '{' and ends with '}'. No preface. No code fences."

JSON_REPAIR_SYSTEM_MESSAGE = "You are a strict JSON fixer. Output ONLY valid JSON. No extra text."