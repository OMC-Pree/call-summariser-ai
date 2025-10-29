# Bedrock Prompt Management Setup Guide

This directory contains templates and scripts for managing prompts in AWS Bedrock Prompt Management.

## Overview

Bedrock Prompt Management allows you to:
- **Version control** your prompts with rollback capability
- **A/B test** different prompt variations
- **Update prompts** without code deployments
- **Track performance** across prompt versions
- **Collaborate** on prompt engineering with your team

## Files

- `summary_prompt_template.json` - Template for call summary generation prompt
- `case_check_prompt_template.json` - Template for compliance case checking prompt
- `create_prompts.py` - Automation script to create prompts via API
- `README.md` - This file

## Quick Start

### Option 1: Automated Creation (Recommended)

Use the Python script to automatically create prompts:

```bash
# Install required dependencies
pip install boto3

# Create all prompts
python create_prompts.py --create all

# Or create individual prompts
python create_prompts.py --create summary
python create_prompts.py --create case-check

# List existing prompts
python create_prompts.py --list

# Get details for a specific prompt
python create_prompts.py --get <prompt-id>
```

The script will:
1. Create the prompts in Bedrock Prompt Management
2. Create initial versions
3. Save ARNs to `prompt_arns.json`
4. Display environment variables to add to your Lambda functions

### Option 2: Manual Creation via AWS Console

If you prefer to use the AWS Console:

1. **Navigate to Bedrock Console**
   - Go to AWS Console → Amazon Bedrock
   - Select "Prompt management" from the left sidebar
   - Click "Create prompt"

2. **Create Summary Prompt**
   - Name: `call-summariser-summary`
   - Description: "Financial coaching call summary generation prompt with structured output via Tool Use"
   - Copy the content from `summary_prompt_template.json`

   **System Prompt:**
   ```
   You are a financial assistant that summarizes coaching calls. Always use the provided tool to structure your response.
   ```

   **User Prompt:**
   ```
   You are a financial assistant summarizing a customer coaching call.

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
   {{transcript}}
   ```

   **Variables:**
   - `transcript` - The call transcript to summarize

   **Model Configuration:**
   - Model: Claude 3.5 Sonnet v2 (`anthropic.claude-3-5-sonnet-20241022-v2:0`)
   - Temperature: 0.3
   - Max tokens: 1200

   **Tools:**
   - Copy the `toolConfiguration` section from `summary_prompt_template.json`
   - This defines the `submit_call_summary` tool with JSON schema

3. **Create Case Check Prompt**
   - Name: `call-summariser-case-check`
   - Description: "Financial coaching call compliance case checking prompt with structured output via Tool Use"
   - Follow similar steps using `case_check_prompt_template.json`

   **Variables:**
   - `meeting_id` - Unique identifier for the meeting
   - `model_version` - Model version identifier
   - `prompt_version` - Prompt version identifier
   - `kb_examples` - Knowledge base examples (optional)
   - `checklist_json` - JSON checklist of compliance checks
   - `cleaned_transcript` - The cleaned transcript to audit

4. **Create Versions**
   - After creating each prompt, click "Create version"
   - Version 1 description: "Initial baseline version"
   - Copy the Version ARN (you'll need this for Lambda configuration)

## Integration with Lambda Functions

After creating prompts, you need to:

1. **Add Environment Variables to Lambda Functions**

   For `summarise` Lambda:
   ```
   PROMPT_ARN_SUMMARY=arn:aws:bedrock:eu-west-2:ACCOUNT:prompt/PROMPT_ID:VERSION
   USE_PROMPT_MANAGEMENT=true
   ```

   For `case_check` Lambda:
   ```
   PROMPT_ARN_CASE_CHECK=arn:aws:bedrock:eu-west-2:ACCOUNT:prompt/PROMPT_ID:VERSION
   USE_PROMPT_MANAGEMENT=true
   ```

2. **Update IAM Permissions**

   Add to Lambda execution role:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "bedrock:GetPrompt",
           "bedrock:InvokeModel"
         ],
         "Resource": [
           "arn:aws:bedrock:eu-west-2:ACCOUNT:prompt/*"
         ]
       }
     ]
   }
   ```

3. **Code changes** will be handled in Phase 2

## Prompt Versioning Workflow

### Creating a New Prompt Version

1. **Test changes locally first**
   ```bash
   # Update the template JSON file with your changes
   # Run evals against the new prompt (see Phase 3)
   ```

2. **Create new version in console or via API**
   - Option A: Use AWS Console → Edit prompt → Save as new version
   - Option B: Use `create_prompts.py` script with updated template

3. **Compare with baseline**
   ```bash
   # Run evals comparing old vs new version
   python ../evals/eval_master.py --compare-prompts \
     --baseline-arn "arn:...:1" \
     --candidate-arn "arn:...:2"
   ```

4. **Deploy to production**
   - Update Lambda environment variable with new version ARN
   - Monitor CloudWatch metrics
   - Rollback if needed by reverting to previous version ARN

## Best Practices

### Prompt Naming Convention
- Use descriptive names: `call-summariser-{function}-{variant}`
- Example: `call-summariser-summary-concise`, `call-summariser-summary-detailed`

### Version Descriptions
- Always include meaningful version descriptions
- Document what changed: "Improved theme detection accuracy", "Added emphasis on compliance language"
- Reference eval results: "Avg score improved from 0.82 to 0.87"

### Variable Naming
- Use clear, descriptive variable names
- Document expected format in variable description
- Use snake_case for consistency

### Testing Strategy
1. **Local testing** - Test prompt changes with your eval dataset
2. **Staging deployment** - Deploy to staging Lambda first
3. **A/B testing** - Run both versions in parallel if critical
4. **Gradual rollout** - Monitor metrics before full deployment

## Troubleshooting

### Common Issues

**Issue: "Prompt not found" error**
- Verify the ARN is correct
- Check the version number is valid
- Ensure Lambda has permission to access the prompt

**Issue: Variables not substituting correctly**
- Verify variable names match exactly (case-sensitive)
- Check that variables are passed in correct format to Converse API
- Use `{{variable_name}}` syntax in prompt template

**Issue: Tool configuration not working**
- Ensure tool schema matches your Pydantic models exactly
- Verify `toolChoice` is set to force tool use
- Check that tool names match between definition and choice

## Next Steps

After completing Phase 1:
- ✅ Prompts created in Bedrock Prompt Management
- ✅ ARNs saved and ready for Lambda configuration

Proceed to:
- **Phase 2**: Update Lambda functions to use Prompt Management API
- **Phase 3**: Extend eval framework for prompt version comparison

## Resources

- [AWS Bedrock Prompt Management Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management.html)
- [Converse API with Prompt Management](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management-code-ex.html)
- [Prompt Engineering Best Practices](https://docs.anthropic.com/claude/docs/prompt-engineering)
