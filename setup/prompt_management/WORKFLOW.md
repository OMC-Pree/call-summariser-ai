# Prompt Management Workflow

## Architecture

```
AWS Parameter Store (SSM)
├── /call-summariser/prompts/summary/current → ARN
├── /call-summariser/prompts/case-check/current → ARN
└── /call-summariser/prompts/vulnerability-assessment/current → ARN
                 ↓
         Lambda (at startup)
         Reads current prompt ARN
                 ↓
    Bedrock Prompt Management
    Fetches prompt text
                 ↓
    Adds tools at runtime
                 ↓
    Calls Converse API
```

**Key Benefit:** Change prompts without code deployment!

---

## Initial Setup

### 1. Create Prompts (One-time)
```bash
cd setup/prompt_management
python create_prompts.py --create all
```

### 2. Initialize Parameter Store (One-time)
```bash
./init_parameters.sh
```

### 3. Deploy Lambda (One-time)
```bash
cd ../..
sam build && sam deploy
```

---

## Daily Workflow: Updating Prompts

### Scenario: You want to improve the summary prompt

**Step 1: Edit prompt in AWS Console**
```
AWS Console → Bedrock → Prompt Management
→ Select "call-summariser-summary-v2"
→ Edit prompt text
→ Save as new version (e.g., version 2)
```

**Step 2: Update Parameter Store** ⭐ **NO CODE DEPLOYMENT NEEDED**
```bash
# Get new version ARN from console, then:
aws ssm put-parameter \
  --name "/call-summariser/prompts/summary/current" \
  --value "arn:aws:bedrock:eu-west-2:ACCOUNT_ID:prompt/576JPUH90Y:2" \
  --overwrite \
  --region eu-west-2
```

**Step 3: Force Lambda cold start** (optional, otherwise happens naturally)
```bash
# Update env var to force restart
aws lambda update-function-configuration \
  --function-name call-summariser-SummariseFunction-XXX \
  --environment Variables={FORCE_RESTART=true}
```

**Done!** Lambda picks up new prompt on next invocation.

---

## A/B Testing Different Prompt Versions

### Scenario: Test prompt v2 in staging, keep v1 in prod

**Staging:**
```bash
aws ssm put-parameter \
  --name "/call-summariser/prompts/summary/staging" \
  --value "arn:...prompt/576JPUH90Y:2" \
  --overwrite

# Update staging Lambda env var:
PROMPT_PARAM_NAME_SUMMARY=/call-summariser/prompts/summary/staging
```

**Production:**
```bash
# Keeps using: /call-summariser/prompts/summary/current → v1
```

---

## Rollback

### Scenario: New prompt has issues, rollback immediately

```bash
# Instant rollback - just point to old version
aws ssm put-parameter \
  --name "/call-summariser/prompts/summary/current" \
  --value "arn:aws:bedrock:eu-west-2:ACCOUNT_ID:prompt/576JPUH90Y:1" \
  --overwrite \
  --region eu-west-2
```

Lambda picks up old prompt on next cold start (usually within minutes).

---

## Viewing Current Prompts

```bash
# Check what prompt is currently active
aws ssm get-parameter \
  --name "/call-summariser/prompts/summary/current" \
  --region eu-west-2

# View all prompt versions
python create_prompts.py --list

# Get specific prompt details
python create_prompts.py --get 576JPUH90Y
```

---

## Best Practices

### 1. Version Your Prompts
- Use descriptive version descriptions in Bedrock Console
- Document what changed: "Improved theme detection", "Fixed compliance language"

### 2. Test Before Production
- Create test parameters: `/call-summariser/prompts/summary/test`
- Test with real data before updating `/current`

### 3. Track Performance
- Log prompt version in CloudWatch
- Compare metrics between versions
- Build evals to quantify improvements

### 4. Keep Parameter Store Clean
- Use clear naming: `/call-summariser/prompts/{type}/current`
- Document in this file when creating new parameters

---

## Troubleshooting

**Problem:** Lambda still using old prompt after parameter update

**Solution:** Force cold start by updating Lambda environment variable or wait 10-15 minutes

**Problem:** Parameter not found error

**Solution:** Run `./init_parameters.sh` to create missing parameters

**Problem:** Access denied to Bedrock prompts

**Solution:** Ensure Lambda has `bedrock:GetPrompt` IAM permission

---

## Parameter Store Reference

| Parameter Name | Description | Example Value |
|----------------|-------------|---------------|
| `/call-summariser/prompts/summary/current` | Active summary prompt ARN | `arn:aws:bedrock:eu-west-2:ACCOUNT_ID:prompt/576JPUH90Y:1` |
| `/call-summariser/prompts/case-check/current` | Active case check prompt ARN | `arn:aws:bedrock:eu-west-2:ACCOUNT_ID:prompt/JFLTWRZ5F4:1` |
| `/call-summariser/prompts/vulnerability-assessment/current` | Active vulnerability assessment prompt ARN | `arn:aws:bedrock:eu-west-2:ACCOUNT_ID:prompt/OQNE6VMD21:1` |
| `/call-summariser/prompts/summary/test` | (Optional) Test prompt ARN | `arn:aws:bedrock:eu-west-2:ACCOUNT_ID:prompt/576JPUH90Y:2` |

---

## Future: Automated Evals

```bash
# Future workflow (when evals are built)
# 1. Create new prompt version in Bedrock Console
# 2. Run evals comparing versions
python evals/compare_prompts.py \
  --baseline "arn:...prompt/576JPUH90Y:1" \
  --candidate "arn:...prompt/576JPUH90Y:2"

# 3. If metrics improve, promote to production
aws ssm put-parameter \
  --name "/call-summariser/prompts/summary/current" \
  --value "arn:...prompt/576JPUH90Y:2" \
  --overwrite
```
