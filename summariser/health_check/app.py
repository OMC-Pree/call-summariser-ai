#!/usr/bin/env python3
"""
Health check endpoint for Call Summarizer system

Provides comprehensive health checks for all system dependencies.
"""

import json
import time
import boto3
from datetime import datetime, timezone
from typing import Dict, Any, List
import logging
import os

# Import error handling
from utils.error_handler import lambda_error_handler
from utils.retry_handler import DynamoDBRetryWrapper, BedrockRetryWrapper, S3RetryWrapper
from constants import *


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checker for all system dependencies"""

    def __init__(self):
        self.checks = []
        self.start_time = time.time()

    def add_check(self, name: str, status: str, response_time_ms: float,
                  details: Dict[str, Any] = None, error: str = None):
        """Add a health check result"""
        self.checks.append({
            'service': name,
            'status': status,  # 'healthy', 'degraded', 'unhealthy'
            'response_time_ms': round(response_time_ms, 2),
            'details': details or {},
            'error': error,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    def check_dynamodb(self) -> None:
        """Check DynamoDB table connectivity and performance"""
        start = time.time()
        try:
            dynamodb_wrapper = DynamoDBRetryWrapper(SUMMARY_JOB_TABLE)

            # Simple connectivity test
            response = dynamodb_wrapper.get_item(
                Key={"meetingId": "health-check-test"},
                ConsistentRead=False
            )

            response_time = (time.time() - start) * 1000

            # Check response time
            if response_time < 100:
                status = 'healthy'
            elif response_time < 500:
                status = 'degraded'
            else:
                status = 'unhealthy'

            self.add_check(
                'dynamodb',
                status,
                response_time,
                details={
                    'table_name': SUMMARY_JOB_TABLE,
                    'region': 'eu-west-2'
                }
            )

        except Exception as e:
            response_time = (time.time() - start) * 1000
            self.add_check(
                'dynamodb',
                'unhealthy',
                response_time,
                error=str(e)
            )

    def check_s3(self) -> None:
        """Check S3 bucket connectivity and permissions"""
        start = time.time()
        try:
            s3_wrapper = S3RetryWrapper()

            # Test bucket access by listing objects with minimal prefix
            test_key = f"health-check/{datetime.now().strftime('%Y/%m/%d')}/test"

            # Try to put a small test object
            s3_wrapper.put_object(
                bucket=SUMMARY_BUCKET,
                key=test_key,
                body="health-check",
                ContentType="text/plain",
                Metadata={'health-check': 'true'}
            )

            response_time = (time.time() - start) * 1000

            # Check response time
            if response_time < 200:
                status = 'healthy'
            elif response_time < 1000:
                status = 'degraded'
            else:
                status = 'unhealthy'

            self.add_check(
                's3',
                status,
                response_time,
                details={
                    'bucket': SUMMARY_BUCKET,
                    'region': 'eu-west-2'
                }
            )

        except Exception as e:
            response_time = (time.time() - start) * 1000
            self.add_check(
                's3',
                'unhealthy',
                response_time,
                error=str(e)
            )

    def check_sqs(self) -> None:
        """Check SQS queue connectivity and attributes"""
        start = time.time()
        try:
            sqs = boto3.client('sqs')

            # Get queue attributes
            response = sqs.get_queue_attributes(
                QueueUrl=os.environ.get("SUMMARY_JOBS_QUEUE", ""),
                AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
            )

            response_time = (time.time() - start) * 1000
            attributes = response.get('Attributes', {})

            visible_messages = int(attributes.get('ApproximateNumberOfMessages', 0))
            processing_messages = int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0))

            # Determine health based on queue backlog
            if visible_messages < 10:
                status = 'healthy'
            elif visible_messages < 50:
                status = 'degraded'
            else:
                status = 'unhealthy'

            self.add_check(
                'sqs',
                status,
                response_time,
                details={
                    'visible_messages': visible_messages,
                    'processing_messages': processing_messages,
                    'queue_url': os.environ.get("SUMMARY_JOBS_QUEUE", "")
                }
            )

        except Exception as e:
            response_time = (time.time() - start) * 1000
            self.add_check(
                'sqs',
                'unhealthy',
                response_time,
                error=str(e)
            )

    def check_bedrock(self) -> None:
        """Check Bedrock model availability"""
        start = time.time()
        try:
            bedrock_wrapper = BedrockRetryWrapper()

            # Simple test with minimal token usage
            test_response = bedrock_wrapper.invoke_model(
                model_id=MODEL_ID,
                body={
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            )

            response_time = (time.time() - start) * 1000

            # Check if we got a valid response
            if test_response and 'content' in test_response:
                if response_time < 2000:
                    status = 'healthy'
                elif response_time < 5000:
                    status = 'degraded'
                else:
                    status = 'unhealthy'
            else:
                status = 'degraded'

            self.add_check(
                'bedrock',
                status,
                response_time,
                details={
                    'model': 'claude-3-sonnet',
                    'region': 'eu-west-2'
                }
            )

        except Exception as e:
            response_time = (time.time() - start) * 1000
            self.add_check(
                'bedrock',
                'unhealthy',
                response_time,
                error=str(e)
            )

    def check_environment(self) -> None:
        """Check environment configuration"""
        start = time.time()

        required_env_vars = [
            'SUMMARY_BUCKET',
            'SUMMARY_JOB_TABLE',
            'SUMMARY_JOBS_QUEUE',
            'MODEL_VERSION',
            'SUMMARY_SCHEMA_VERSION'
        ]

        missing_vars = []
        for var in required_env_vars:
            if not os.environ.get(var):
                missing_vars.append(var)

        response_time = (time.time() - start) * 1000

        if not missing_vars:
            status = 'healthy'
            details = {'all_required_vars_present': True}
            error = None
        else:
            status = 'unhealthy'
            details = {'missing_variables': missing_vars}
            error = f"Missing required environment variables: {', '.join(missing_vars)}"

        self.add_check(
            'environment',
            status,
            response_time,
            details=details,
            error=error
        )

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status"""
        logger.info("Starting comprehensive health check")

        # Run all checks
        self.check_environment()
        self.check_dynamodb()
        self.check_s3()
        self.check_sqs()

        # Skip Bedrock check in development to avoid costs
        if os.environ.get("AWS_SAM_LOCAL") != "true":
            self.check_bedrock()

        # Calculate overall status
        statuses = [check['status'] for check in self.checks]

        if all(status == 'healthy' for status in statuses):
            overall_status = 'healthy'
        elif any(status == 'unhealthy' for status in statuses):
            overall_status = 'unhealthy'
        else:
            overall_status = 'degraded'

        # Calculate total response time
        total_response_time = (time.time() - self.start_time) * 1000

        return {
            'status': overall_status,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_response_time_ms': round(total_response_time, 2),
            'checks': self.checks,
            'summary': {
                'total_services': len(self.checks),
                'healthy_services': len([c for c in self.checks if c['status'] == 'healthy']),
                'degraded_services': len([c for c in self.checks if c['status'] == 'degraded']),
                'unhealthy_services': len([c for c in self.checks if c['status'] == 'unhealthy'])
            }
        }

@lambda_error_handler()
def lambda_handler(event, context):
    """Health check Lambda handler"""
    logger.info("Health check requested")

    # Check if this is a detailed health check or simple ping
    query_params = event.get('queryStringParameters') or {}
    check_type = query_params.get('type', 'simple')

    if check_type == 'simple':
        # Simple ping response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache'
            },
            'body': json.dumps({
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Service is running'
            })
        }

    else:
        # Comprehensive health check
        health_checker = HealthChecker()
        health_status = health_checker.run_all_checks()

        # Return appropriate HTTP status code
        if health_status['status'] == 'healthy':
            status_code = 200
        elif health_status['status'] == 'degraded':
            status_code = 200  # Still operational
        else:
            status_code = 503  # Service unavailable

        logger.info(f"Health check completed: {health_status['status']}")

        return {
            'statusCode': status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache'
            },
            'body': json.dumps(health_status, indent=2)
        }

# Utility function for manual testing
def test_health_check():
    """Test health check functionality"""
    health_checker = HealthChecker()
    result = health_checker.run_all_checks()

    print("Health Check Results:")
    print("=" * 50)
    print(f"Overall Status: {result['status']}")
    print(f"Total Response Time: {result['total_response_time_ms']:.2f}ms")
    print("\nService Details:")

    for check in result['checks']:
        status_emoji = {'healthy': '✅', 'degraded': '⚠️', 'unhealthy': '❌'}.get(check['status'], '❓')
        print(f"{status_emoji} {check['service']}: {check['status']} ({check['response_time_ms']:.2f}ms)")
        if check['error']:
            print(f"    Error: {check['error']}")

    return result

if __name__ == "__main__":
    test_health_check()