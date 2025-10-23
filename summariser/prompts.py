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

Analyze the transcript and use the submit_call_summary tool to provide:
- A concise summary of the call
- Key discussion points (as an array of strings)
- Action items with descriptions
- Sentiment analysis (Positive/Neutral/Negative with confidence)
- Identified themes (0-7 themes from: Budgeting, ISA, Mortgage, Pension, Protection, Debt, etc.)

Rules:
- Use British English and GBP where relevant
- Only include themes clearly supported by the transcript
- Provide short evidence quotes when possible

Transcript:
{transcript}"""

# Case Check Prompt Template
CASE_CHECK_PROMPT_TEMPLATE = """You are auditing a financial coaching call against a checklist.

Use the submit_case_check tool to provide your assessment.

STATUS DEFINITIONS:
- "Competent": The coach fully met the requirement with clear evidence. This is the standard for good performance.
- "CompetentWithDevelopment": The coach met the requirement but there are areas for improvement. The core requirement was satisfied but execution could be better.
- "Fail": The requirement was missed, incorrect, explicitly prohibited, or violated compliance rules.
- "NotApplicable": This check does not apply to this specific session (e.g., client is not over 50 for age-specific checks).
- "Inconclusive": Insufficient evidence to make a clear judgment.

ASSESSMENT RULES:
- Use 'Competent' when the coach clearly fulfilled the requirement with good quality evidence.
- Use 'CompetentWithDevelopment' when the requirement was met but could be improved (e.g., brief confirmation instead of thorough discussion).
- Use 'Fail' when the requirement was missed, incorrect, or explicitly prohibited (like giving regulated advice or steering).
- Use 'NotApplicable' if the check does not apply to this session.
- Use 'Inconclusive' only if there's genuinely not enough evidence to make a judgment.

EVIDENCE & COMMENTS:
- Always provide specific evidence quotes from the transcript when available.
- Evidence quotes should be actual dialogue excerpts, not summaries.
- Comments should explain WHY you assigned the status and what could be improved (for CompetentWithDevelopment) or what went wrong (for Fail).
- Set 'evidence_quote' and 'comment' to empty string ('') only if truly unavailable.

COMPLIANCE VIOLATIONS:
- If the coach gives regulated financial advice (specific product recommendations) → 'Fail'
- If the coach uses steering language ("the best route", "you should", "I would do") → 'Fail'
- If high-severity compliance items are missed → 'Fail'

METADATA:
- Set meeting_id EXACTLY to "{meeting_id}".
- Set model_version EXACTLY to "{model_version}".
- Set prompt_version EXACTLY to "{prompt_version}".
- "evidence_quote" and "comment" must be strings; if unavailable use "" (empty string), not null.
- If any other value is unknown, use null (not placeholders like "[REDACTED]").

{kb_examples}

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
SUMMARY_SYSTEM_MESSAGE = "You are a financial assistant that summarizes coaching calls. Always use the provided tool to structure your response."

CASE_CHECK_SYSTEM_MESSAGE = "You are a compliance auditor. Always use the provided tool to structure your assessment results."

JSON_REPAIR_SYSTEM_MESSAGE = "You are a strict JSON fixer. Output ONLY valid JSON. No extra text."