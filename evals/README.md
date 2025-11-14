# Vulnerability Ground Truth Collection

This directory contains the infrastructure for collecting validated ground truth labels from head coaches for vulnerability detection model evaluation.

## Files

```
evals/
├── README.md                      # This file
├── meeting_level_review.html     # Coach review interface for collecting ground truth reviews
├── generate_ai_responses.py      # Script to process meetings through multiple AI models
├── evaluate_models.py             # Script to evaluate model performance vs ground truth
├── requirements.txt               # Python dependencies for the scripts
├── .env.example                   # Environment variables template
├── .env                           # Your API keys (git-ignored)
└── .gitignore                     # Ensures .env is never committed
```

## Quick Start

### 1. Deploy Infrastructure
```bash
cd /Users/pree/Documents/AI\ POCs/call-summariser/call-summariser
sam build
sam deploy
```

This creates:
- DynamoDB table: `vulnerability-assessments` (stores third-party + AI assessments)
- DynamoDB table: `vulnerability-ground-truth-reviews` (stores coach reviews)
- API Gateway endpoint: `GET /reviews/pending`
- API Gateway endpoint: `POST /review`
- Lambda functions for fetching and saving reviews

### 2. Generate AI Model Responses

Install dependencies:
```bash
cd evals
pip install -r requirements.txt
```

Set up environment variables (create `.env` file):
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Your `.env` file should contain:
```
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
AWS_REGION=eu-west-2
S3_BUCKET=call-summariser-summaries
```

Process all pending meetings through AI models:
```bash
python3 generate_ai_responses.py --limit 10
```

Process a specific meeting:
```bash
python3 generate_ai_responses.py --meeting-id 91085608879
```

This script:
- Fetches transcripts from S3
- Calls GPT-4o, Gemini 2.5 Flash, and Claude 3 Sonnet
- Stores AI responses with reasoning in DynamoDB `vulnerability-assessments` table

### 3. Open Review Interface

**Option A: Use S3-hosted version (Recommended)**
```
http://coach-review-interface-vulnerability.s3-website.eu-west-2.amazonaws.com
```

**Option B: Open local file**
```bash
open evals/meeting_level_review.html
```

**Deploy updates to S3:**
```bash
cd evals
./deploy_to_s3.sh
```

Or manually:
```bash
aws s3 cp evals/meeting_level_review.html s3://coach-review-interface-vulnerability/index.html --content-type "text/html"
```

**Optional - Enable AI Model Comparisons:**

By default, the AI model responses are hidden to avoid biasing coach reviews. To enable the "🤖 Show AI Model Responses" button:

1. Open `evals/meeting_level_review.html` in a text editor
2. Find the `FEATURE_FLAGS` section at the top of the `<script>` tag (around line 337)
3. Change `SHOW_AI_MODELS: false` to `SHOW_AI_MODELS: true`
4. Save the file
5. Redeploy to S3: `cd evals && ./deploy_to_s3.sh`
6. Refresh the browser page

### 4. Review Process
1. Enter your coach email
2. Review each meeting's third-party assessment
3. **Optional** (if feature flag enabled): Click "🤖 Show AI Model Responses" to see GPT-4o, Gemini 2.5 Flash, and Claude 3 Sonnet assessments with reasoning
4. Click "Agree" or "Disagree" (with corrections)
5. Reviews automatically save to DynamoDB

### 5. Evaluate Model Performance

After coaches complete their reviews, evaluate AI model performance:

```bash
python3 evaluate_models.py --html
```

This generates:
- Console output with performance metrics
- `evaluation_results.json` - Detailed results in JSON format
- `evaluation_results.html` - Visual report with charts and comparisons

The evaluation compares:
- GPT-4o vs ground truth
- Gemini 2.5 Flash vs ground truth
- Claude 3 Sonnet vs ground truth
- Third-party (Aveni) baseline vs ground truth

Metrics calculated:
- **Exact Accuracy**: % of predictions matching ground truth exactly
- **Within-1 Accuracy**: % of predictions within 1 rating level
- **Mean Absolute Error**: Average difference from ground truth
- **Over/Under-predictions**: Bias analysis

## Architecture

```
Third-party assessments → DynamoDB (vulnerability-assessments)
    ↓
generate_ai_responses.py → Fetch transcripts from S3
    ↓
Call GPT-4o, Gemini 2.5 Flash, Claude 3 Sonnet → Store ai_responses in DynamoDB
    ↓
GET /reviews/pending → Fetch assessments with AI responses
    ↓
meeting_level_review.html → Coach reviews (view AI responses, agree/disagree)
    ↓
POST /review → Update DynamoDB with coach decision
    ↓
Ground truth data available for model evaluation
```

## Infrastructure

| Component | Name | Region |
|-----------|------|--------|
| DynamoDB Table | `vulnerability-ground-truth-reviews` | us-east-1 |
| Lambda Function | `SaveVulnerabilityGroundTruthReviewFunction` | us-east-1 |
| API Gateway | Auto-generated REST API | us-east-1 |

## Documentation

See main project documentation for architecture overview.

## Data Schema

### Assessment Record (DynamoDB: vulnerability-assessments)
```json
{
  "meeting_id": "92125626617",
  "assessment_id": "third-party",
  "vulnerability_rating": "High/4",
  "vulnerability_types": ["Capability: Learning Difficulties", "Life Events: Bereavement"],
  "evidence_quotes": ["quote from transcript..."],
  "review_status": "pending",
  "assessed_at": "2025-11-13T10:00:00Z",
  "ai_responses": {
    "gpt-4o": {
      "rating": "Medium/3",
      "vulnerability_types": ["Health: Mental Health Condition", "Resilience: Low Emotional Resilience"],
      "reasoning": "The client has been diagnosed with ADHD and autism..."
    },
    "gemini-2.5-flash": {
      "rating": "Medium/3",
      "vulnerability_types": ["Health: Mental Health Condition", "Resilience: Low Emotional Resilience", "Capability: Low Confidence Financial Matters"],
      "reasoning": "The client presents as highly proactive but diagnosed with ADHD and autism..."
    },
    "claude-3-sonnet": {
      "rating": "Medium/3",
      "vulnerability_types": ["Health: Mental Health Condition", "Resilience: Low Emotional Resilience"],
      "reasoning": "The client mentions being diagnosed with ADHD and autism within the past two years..."
    }
  }
}
```

### Review Record (DynamoDB)
```json
{
  "review_id": "92125626617#coach@example.com",
  "meeting_id": "92125626617",
  "coach_email": "coach@example.com",
  "action": "correct",
  "third_party_rating": "High/4",
  "third_party_types": "Capability: Learning Difficulties; Life Events: Bereavement",
  "corrected_rating": "Medium/3",
  "corrected_types": "Life Events: Bereavement",
  "reasoning": "Evidence only supports bereavement, not learning difficulties",
  "timestamp": "2025-11-13T10:30:00.000Z"
}
```

### Ground Truth (Output)
```
meeting_id|vulnerability_rating|vulnerability_types|reviewed_by|action
92125626617|Medium/3|Life Events: Bereavement|coach@example.com|correct
```

## Status

✅ **Production Ready**

- Meeting data embedded in HTML interface
- DynamoDB storage configured
- API endpoint deployed
- Review interface functional

---

**Last Updated**: 2025-11-13
