#!/bin/bash
# Deploy the review interface to S3

set -e

BUCKET="coach-review-interface-vulnerability"
HTML_FILE="meeting_level_review.html"

echo "🚀 Deploying review interface to S3..."

# Upload HTML file
aws s3 cp "${HTML_FILE}" "s3://${BUCKET}/index.html" \
    --content-type "text/html" \
    --cache-control "max-age=300"

echo "✅ Deployment complete!"
echo ""
echo "🌐 Website URL:"
echo "   http://${BUCKET}.s3-website.eu-west-2.amazonaws.com"
echo ""
echo "📝 To enable AI model comparisons:"
echo "   1. Edit ${HTML_FILE}"
echo "   2. Change SHOW_AI_MODELS: false to SHOW_AI_MODELS: true"
echo "   3. Run this script again to deploy"
