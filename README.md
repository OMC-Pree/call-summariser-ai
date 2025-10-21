# Call Summariser AI

## Overview

Call Summariser AI is an automated system that processes business call recordings and produces structured summaries enriched with insights and compliance checks.  The service records calls or ingests transcripts, redacts personally identifiable information (PII), converts the speech to text (if necessary), and uses large‑language‑model (LLM) analysis to generate rich summaries.  Each summary includes:

* **Key points and themes** – the main discussion topics extracted from the call.
* **Action items** – follow‑up tasks for participants.
* **Sentiment and quality scores** – indications of how the call went.
* **Compliance checks** – optional rules that evaluate whether the call meets business or regulatory requirements.

A web‑based dashboard (not yet public) is used for analytics, version comparison and further insights.

## Process Flow

The system is built on AWS using Lambda functions, AWS Step Functions and DynamoDB.  A high‑level overview of the processing pipeline is:

1. **Initiate summary (`POST /summarise`)** – Clients submit a request with a `meetingId` and either a transcript or a Zoom meeting ID.  The API validates the request, stores a job record in DynamoDB and starts a Step Functions workflow execution.
2. **Step Functions workflow orchestration** – A state machine coordinates the following steps:
   * **Fetch transcript** – Retrieves the transcript from the request or fetches it from Zoom via their API.
   * **Normalize roles** – Standardizes speaker labels (e.g., COACH and CLIENT).
   * **PII detection & redaction** – Uses Amazon Comprehend to detect and redact personally identifiable information.
   * **Summarize** – Calls a Bedrock LLM to generate a structured summary with themes, action items and sentiment analysis.
   * **Validate & repair** – Validates the LLM output JSON and attempts repairs if needed.
   * **Case check (optional)** – If enabled, runs a second LLM prompt that applies compliance rules in parallel with summarization.
   * **Persist summary** – Saves the validated summary to S3 and updates DynamoDB.
   * **A2I review (optional)** – If the pass‑rate falls below a threshold or there is a high‑severity failure, initiates a human review loop via Amazon A2I.
   * **Update status** – Updates the final job status in DynamoDB.
3. **Human review (optional)** – For flagged jobs, the system initiates a human review loop via Amazon A2I.  A separate poller Lambda monitors completion and updates the job status back to `COMPLETED` along with the reviewer's decision and comments.
4. **Retrieve results** – Clients can:
   * List completed summaries with `GET /summaries`.
   * Check the status of a job and fetch a pre‑signed summary JSON with `GET /status?meetingId`.
   * Download case‑check reports with `GET /case?meetingId`.

Summaries and case‑check reports are stored in Amazon S3, and object keys are recorded in DynamoDB for retrieval.

## API Endpoints

| Endpoint & Method | Description |
| --- | --- |
| `POST /summarise` | Create a new summarisation job for a meeting ID and transcript or Zoom recording ID.  Returns a job identifier. |
| `GET /summaries` | List existing summary jobs and their statuses. |
| `GET /status?meetingId=<id>` | Retrieve the current status of a job and, if completed, obtain a pre‑signed URL for the summary JSON. |
| `GET /case?meetingId=<id>` | Retrieve a pre‑signed URL for the case‑check report (when case checking is enabled). |

## Output Format

The summary endpoint returns structured JSON similar to the following (fields omitted for brevity):

```json
{
  "summary_schema_version": "1.2",
  "model_version": "bedrock:claude-3-sonnet-20240229",
  "prompt_version": "2025-09-22-a",
  "meeting": {
    "id": "<meetingId>",
    "employerName": "<employer>",
    "coach": "<coach name>",
    "createdAt": "<ISO timestamp>"
  },
  "themes": [
    {
      "id": "<theme id>",
      "label": "<label>",
      "group": "<group>",
      "confidence": 0.8,
      "evidence_quote": "<quote from call>"
    },
    …
  ],
  "summary": "Concise paragraph summarising the call…",
  "actions": [
    {
      "id": "A1",
      "text": "<action item>"
    },
    …
  ],
  "call_metadata": {
    "source": "zoom_api",
    "saved_at": "<ISO timestamp>",
    "insights_version": "2025-08-30-a",
    "schema_version": "1.2"
  },
  "insights": {
    "action_count": 3,
    "theme_count": 3,
    "sentiment_label": "Positive",
    "is_escalation_candidate": false,
    "quality_score": 0.67
  }
}
```

Case‑check results include a checklist of compliance tests with statuses (`Pass`, `NotApplicable`, etc.), evidence spans and explanatory comments, along with an overall pass rate and any high‑severity flags.

## Roadmap

The project is actively evolving.  Upcoming tasks include prompt and schema version management, repository cleanup, extending case‑check coverage, implementing retrieval‑augmented summarisation, prototyping real‑time summarisation, building a continuous evaluation pipeline and standardising logging.

## Technology Stack

This project uses:

* **AWS Lambda** – serverless functions for the API layer, summary processing and polling.
* **AWS Step Functions** – orchestrates the workflow for transcript processing, summarization and compliance checks.
* **Amazon DynamoDB** – durable storage for job metadata and object keys.
* **Amazon S3** – storage for summaries and case‑check reports.
* **Bedrock (Claude)** – large language model for summarisation and compliance prompting.
* **Amazon A2I** – optional human‑in‑the‑loop review of flagged summaries.

## Contributing

Contributions are welcome!  Please fork the repository, create a feature branch and open a pull request describing your changes.  For significant changes or new functionality, please open an issue first to discuss the proposal.
