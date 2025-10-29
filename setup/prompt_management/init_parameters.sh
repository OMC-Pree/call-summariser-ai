#!/bin/bash
# Initialize AWS Systems Manager Parameter Store with prompt ARNs
# Run this after creating prompts with create_prompts.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARNS_FILE="$SCRIPT_DIR/prompt_arns.json"
AWS_REGION="${AWS_REGION:-eu-west-2}"

echo "🔧 Initializing Parameter Store with prompt ARNs..."
echo ""

# Check if prompt_arns.json exists
if [ ! -f "$ARNS_FILE" ]; then
    echo "❌ Error: prompt_arns.json not found"
    echo "   Run 'python create_prompts.py --create all' first"
    exit 1
fi

# Extract ARNs from JSON
SUMMARY_ARN=$(cat "$ARNS_FILE" | python3 -c "import sys, json; print(json.load(sys.stdin)['summary']['version_arn'])")
CASE_CHECK_ARN=$(cat "$ARNS_FILE" | python3 -c "import sys, json; print(json.load(sys.stdin)['case_check']['version_arn'])")

echo "📋 Found prompt ARNs:"
echo "   Summary: $SUMMARY_ARN"
echo "   Case Check: $CASE_CHECK_ARN"
echo ""

# Create/update parameters
echo "📝 Creating Parameter Store parameters..."

aws ssm put-parameter \
    --name "/call-summariser/prompts/summary/current" \
    --value "$SUMMARY_ARN" \
    --type "String" \
    --description "Current prompt ARN for summary generation" \
    --overwrite \
    --region "$AWS_REGION"

echo "✅ Created: /call-summariser/prompts/summary/current"

aws ssm put-parameter \
    --name "/call-summariser/prompts/case-check/current" \
    --value "$CASE_CHECK_ARN" \
    --type "String" \
    --description "Current prompt ARN for case check" \
    --overwrite \
    --region "$AWS_REGION"

echo "✅ Created: /call-summariser/prompts/case-check/current"

echo ""
echo "✅ Parameter Store initialization complete!"
echo ""
echo "📋 To view parameters:"
echo "   aws ssm get-parameter --name /call-summariser/prompts/summary/current --region $AWS_REGION"
echo "   aws ssm get-parameter --name /call-summariser/prompts/case-check/current --region $AWS_REGION"
echo ""
echo "🔄 To update to a new prompt version (no code deployment needed):"
echo "   aws ssm put-parameter \\"
echo "     --name /call-summariser/prompts/summary/current \\"
echo "     --value \"arn:aws:bedrock:$AWS_REGION:ACCOUNT:prompt/ID:NEW_VERSION\" \\"
echo "     --overwrite --region $AWS_REGION"
echo ""
echo "💡 Lambda will pick up the new prompt on next cold start (usually within minutes)"
