#!/bin/bash

# Production deployment script for Call Summarizer
# This script deploys the main application with monitoring and health checks

set -e  # Exit on any error

echo "🚀 Starting production deployment for Call Summarizer"

# Configuration
STACK_NAME=${STACK_NAME:-call-summariser}
ENVIRONMENT=${ENVIRONMENT:-prod}
REGION=${AWS_DEFAULT_REGION:-eu-west-2}
NOTIFICATION_EMAIL=${NOTIFICATION_EMAIL:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    print_error "AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if SAM CLI is installed
if ! command -v sam &> /dev/null; then
    print_error "SAM CLI not installed. Please install it first: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html"
    exit 1
fi

print_success "Prerequisites check passed"

# Step 1: Run tests
print_status "Running unit tests..."
if [[ -f "tests/run_tests.py" ]]; then
    cd tests
    python run_tests.py
    if [[ $? -ne 0 ]]; then
        print_error "Tests failed. Aborting deployment."
        exit 1
    fi
    cd ..
    print_success "All tests passed"
else
    print_warning "No tests found, skipping test phase"
fi

# Step 2: Build application
print_status "Building SAM application..."
sam build --use-container
if [[ $? -ne 0 ]]; then
    print_error "SAM build failed"
    exit 1
fi
print_success "SAM build completed"

# Step 3: Deploy main application
print_status "Deploying main application stack..."
sam deploy \
    --stack-name $STACK_NAME \
    --capabilities CAPABILITY_IAM \
    --region $REGION \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset

if [[ $? -ne 0 ]]; then
    print_error "Main application deployment failed"
    exit 1
fi
print_success "Main application deployed"

# Step 4: Deploy monitoring stack
print_status "Deploying monitoring and alerting..."
aws cloudformation deploy \
    --template-file monitoring/cloudwatch-template.yaml \
    --stack-name "${STACK_NAME}-monitoring" \
    --parameter-overrides \
        StackName=$STACK_NAME \
        Environment=$ENVIRONMENT \
        NotificationEmail="$NOTIFICATION_EMAIL" \
    --capabilities CAPABILITY_IAM \
    --region $REGION

if [[ $? -ne 0 ]]; then
    print_error "Monitoring deployment failed"
    exit 1
fi
print_success "Monitoring stack deployed"

# Step 5: Get outputs and display URLs
print_status "Retrieving deployment information..."

# Get API Gateway URL
API_URL=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
    --output text)

# Get Dashboard URL
DASHBOARD_URL=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}-monitoring" \
    --region $REGION \
    --query "Stacks[0].Outputs[?OutputKey=='DashboardURL'].OutputValue" \
    --output text 2>/dev/null || echo "Dashboard URL not available")

# Step 6: Run post-deployment health check
print_status "Running post-deployment health check..."

if [[ ! -z "$API_URL" ]]; then
    # Test the health endpoint
    HEALTH_URL="${API_URL%/summarise}/health?type=detailed"

    print_status "Testing health endpoint: $HEALTH_URL"

    HEALTH_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/health_response.json "$HEALTH_URL" || echo "000")

    if [[ "$HEALTH_RESPONSE" == "200" ]]; then
        print_success "Health check passed"

        # Show health summary
        if command -v jq &> /dev/null; then
            echo "Health Summary:"
            jq '.summary' /tmp/health_response.json 2>/dev/null || echo "Health data available in /tmp/health_response.json"
        fi
    else
        print_warning "Health check returned status: $HEALTH_RESPONSE"
        print_warning "Check the detailed response in /tmp/health_response.json"
    fi
fi

# Step 7: Display deployment summary
echo ""
echo "🎉 Production deployment completed successfully!"
echo "=================================================================="
echo "📊 Deployment Summary:"
echo "  Stack Name: $STACK_NAME"
echo "  Environment: $ENVIRONMENT"
echo "  Region: $REGION"
echo ""
echo "🔗 Important URLs:"
if [[ ! -z "$API_URL" ]]; then
    echo "  API Endpoint: $API_URL"
    echo "  Health Check: ${API_URL%/summarise}/health"
fi
if [[ ! -z "$DASHBOARD_URL" && "$DASHBOARD_URL" != "Dashboard URL not available" ]]; then
    echo "  CloudWatch Dashboard: $DASHBOARD_URL"
fi
echo ""
echo "📋 Next Steps:"
echo "  1. Test the API endpoints with real data"
echo "  2. Monitor the CloudWatch dashboard for metrics"
echo "  3. Set up regular health checks in your monitoring system"
echo "  4. Review and adjust CloudWatch alarms as needed"

if [[ ! -z "$NOTIFICATION_EMAIL" ]]; then
    echo "  5. Check your email ($NOTIFICATION_EMAIL) for SNS subscription confirmation"
fi

echo ""
echo "🛠️  Useful Commands:"
echo "  View logs: sam logs --stack-name $STACK_NAME --tail"
echo "  Run local tests: cd evals && python simple_eval.py --dataset all"
echo "  Monitor metrics: aws cloudwatch get-dashboard --dashboard-name ${STACK_NAME}-monitoring-${ENVIRONMENT}"

# Clean up temporary files
rm -f /tmp/health_response.json

print_success "Deployment script completed"