#!/usr/bin/env python3
"""
Comprehensive error handling framework for production Lambda functions

Provides standardized error handling, logging, and recovery mechanisms.
"""

import json
import traceback
import time
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime, timezone
from functools import wraps
from enum import Enum
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import logging

class ErrorSeverity(Enum):
    """Error severity levels for proper escalation"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ErrorCategory(Enum):
    """Error categories for monitoring and alerting"""
    USER_INPUT = "USER_INPUT"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONFIGURATION = "CONFIGURATION"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    BUSINESS_LOGIC = "BUSINESS_LOGIC"

class CallSummarizerError(Exception):
    """Base exception for call summarizer errors"""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.INTERNAL_ERROR,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, details: Optional[Dict] = None,
                 correlation_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.correlation_id = correlation_id
        self.timestamp = datetime.now(timezone.utc).isoformat()

class ValidationError(CallSummarizerError):
    """Input validation errors"""
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(message, ErrorCategory.USER_INPUT, ErrorSeverity.LOW, **kwargs)
        self.field = field

class ExternalServiceError(CallSummarizerError):
    """External service failures (Bedrock, S3, etc.)"""
    def __init__(self, message: str, service: str, **kwargs):
        super().__init__(message, ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.HIGH, **kwargs)
        self.service = service

class BusinessLogicError(CallSummarizerError):
    """Business logic violations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.MEDIUM, **kwargs)

class ErrorHandler:
    """Centralized error handling and logging"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.cloudwatch = None

        try:
            self.cloudwatch = boto3.client('cloudwatch')
        except Exception:
            self.logger.warning("CloudWatch client not available - metrics disabled")

    def handle_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle and log errors with proper categorization"""

        correlation_id = self._get_correlation_id(context)

        if isinstance(error, CallSummarizerError):
            return self._handle_known_error(error, correlation_id)
        else:
            return self._handle_unknown_error(error, context, correlation_id)

    def _handle_known_error(self, error: CallSummarizerError, correlation_id: str) -> Dict[str, Any]:
        """Handle known application errors"""

        error_data = {
            'error_id': f"ERR_{int(time.time())}",
            'message': error.message,
            'category': error.category.value,
            'severity': error.severity.value,
            'details': error.details,
            'correlation_id': correlation_id,
            'timestamp': error.timestamp
        }

        # Log based on severity
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.logger.error("Application error", extra={'error_data': error_data})
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Application warning", extra={'error_data': error_data})
        else:
            self.logger.info("Application info", extra={'error_data': error_data})

        # Send metrics to CloudWatch
        self._send_error_metric(error.category.value, error.severity.value)

        return self._format_error_response(error_data)

    def _handle_unknown_error(self, error: Exception, context: Optional[Dict],
                            correlation_id: str) -> Dict[str, Any]:
        """Handle unexpected errors"""

        # Categorize common error types
        category, severity = self._categorize_error(error)

        error_data = {
            'error_id': f"ERR_{int(time.time())}",
            'message': str(error),
            'error_type': type(error).__name__,
            'category': category.value,
            'severity': severity.value,
            'traceback': traceback.format_exc(),
            'context': context or {},
            'correlation_id': correlation_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        self.logger.error("Unhandled error", extra={'error_data': error_data})
        self._send_error_metric(category.value, severity.value)

        return self._format_error_response(error_data, include_traceback=False)

    def _categorize_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Categorize unknown errors based on type"""

        if isinstance(error, (ClientError, BotoCoreError)):
            return ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorCategory.USER_INPUT, ErrorSeverity.LOW
        elif isinstance(error, KeyError):
            return ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM
        elif isinstance(error, MemoryError):
            return ErrorCategory.RESOURCE_LIMIT, ErrorSeverity.CRITICAL
        else:
            return ErrorCategory.INTERNAL_ERROR, ErrorSeverity.HIGH

    def _send_error_metric(self, category: str, severity: str):
        """Send error metrics to CloudWatch"""
        if not self.cloudwatch:
            return

        try:
            self.cloudwatch.put_metric_data(
                Namespace='CallSummarizer/Errors',
                MetricData=[
                    {
                        'MetricName': 'ErrorCount',
                        'Dimensions': [
                            {'Name': 'Category', 'Value': category},
                            {'Name': 'Severity', 'Value': severity}
                        ],
                        'Value': 1,
                        'Unit': 'Count',
                        'Timestamp': datetime.now(timezone.utc)
                    }
                ]
            )
        except Exception as e:
            self.logger.warning(f"Failed to send CloudWatch metric: {e}")

    def _get_correlation_id(self, context: Optional[Dict]) -> str:
        """Extract or generate correlation ID"""
        if context and 'correlation_id' in context:
            return context['correlation_id']
        elif context and 'aws_request_id' in context:
            return context['aws_request_id']
        else:
            return f"corr_{int(time.time())}"

    def _format_error_response(self, error_data: Dict, include_traceback: bool = False) -> Dict[str, Any]:
        """Format error response for API"""

        response = {
            'error': {
                'error_id': error_data['error_id'],
                'message': error_data['message'],
                'category': error_data['category'],
                'timestamp': error_data['timestamp'],
                'correlation_id': error_data['correlation_id']
            }
        }

        # Include additional details for internal errors (not user-facing)
        if include_traceback and error_data.get('traceback'):
            response['error']['traceback'] = error_data['traceback']

        return response

def lambda_error_handler(correlation_id_field: str = 'aws_request_id'):
    """Decorator for Lambda function error handling"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
            error_handler = ErrorHandler()

            # Extract correlation ID from context
            correlation_id = getattr(context, correlation_id_field, f"req_{int(time.time())}")
            context_dict = {
                'correlation_id': correlation_id,
                'function_name': getattr(context, 'function_name', 'unknown'),
                'aws_request_id': getattr(context, 'aws_request_id', None)
            }

            try:
                # Add correlation ID to all log messages
                logger = logging.getLogger()
                logger.info(f"Function invoked", extra={
                    'correlation_id': correlation_id,
                    'event_keys': list(event.keys()) if isinstance(event, dict) else 'non-dict'
                })

                result = func(event, context)

                logger.info(f"Function completed successfully", extra={
                    'correlation_id': correlation_id
                })

                return result

            except Exception as e:
                error_response = error_handler.handle_error(e, context_dict)

                # Detect invocation source: Step Functions vs API Gateway
                # Step Functions events don't have 'httpMethod' or 'requestContext'
                is_api_gateway = isinstance(event, dict) and (
                    'httpMethod' in event or
                    'requestContext' in event or
                    'headers' in event
                )

                if is_api_gateway:
                    # Return HTTP response format for API Gateway
                    status_code = 400 if isinstance(e, ValidationError) else 500
                    return {
                        'statusCode': status_code,
                        'headers': {
                            'Content-Type': 'application/json',
                            'X-Correlation-ID': correlation_id
                        },
                        'body': json.dumps(error_response)
                    }
                else:
                    # For Step Functions, raise the error so it can be caught/retried
                    raise

        return wrapper
    return decorator

class InputValidator:
    """Input validation utilities"""

    @staticmethod
    def validate_required_fields(data: Dict, required_fields: list, context: str = "input") -> None:
        """Validate required fields are present"""
        missing_fields = [field for field in required_fields if not data.get(field)]

        if missing_fields:
            raise ValidationError(
                f"Missing required fields in {context}: {', '.join(missing_fields)}",
                details={'missing_fields': missing_fields, 'context': context}
            )

    @staticmethod
    def validate_string_field(value: Any, field_name: str, min_length: int = 1,
                            max_length: int = 10000) -> str:
        """Validate string field with length constraints"""
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field=field_name)

        value = value.strip()

        if len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters",
                field=field_name
            )

        if len(value) > max_length:
            raise ValidationError(
                f"{field_name} must be no more than {max_length} characters",
                field=field_name
            )

        return value

    @staticmethod
    def validate_meeting_id(meeting_id: Any) -> str:
        """Validate meeting ID format"""
        if not isinstance(meeting_id, str):
            raise ValidationError("meetingId must be a string", field="meetingId")

        meeting_id = meeting_id.strip()

        # Updated format validation to allow alphanumeric meeting IDs
        # Allow letters, numbers, hyphens, and underscores
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', meeting_id) or len(meeting_id) < 3 or len(meeting_id) > 50:
            raise ValidationError(
                "meetingId must be 3-50 alphanumeric characters (letters, numbers, hyphens, underscores only)",
                field="meetingId",
                details={'provided_value': meeting_id}
            )

        return meeting_id

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input to prevent injection attacks"""
        if not isinstance(text, str):
            return ""

        # Remove potential script tags and other dangerous patterns
        import re

        # Remove script tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove other potentially dangerous tags
        dangerous_tags = ['iframe', 'object', 'embed', 'link', 'meta']
        for tag in dangerous_tags:
            text = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Limit length to prevent DoS
        if len(text) > 100000:  # 100KB limit
            text = text[:100000]

        return text.strip()

# Utility functions for common error scenarios
def handle_bedrock_error(error: Exception, model_name: str, correlation_id: str = None) -> None:
    """Handle Bedrock-specific errors with proper categorization"""
    if isinstance(error, ClientError):
        error_code = error.response.get('Error', {}).get('Code', 'Unknown')

        if error_code in ['ThrottlingException', 'ServiceQuotaExceededException']:
            raise ExternalServiceError(
                f"Bedrock rate limit exceeded for model {model_name}",
                service="bedrock",
                details={'error_code': error_code, 'model': model_name},
                correlation_id=correlation_id
            )
        elif error_code in ['ValidationException']:
            raise BusinessLogicError(
                f"Invalid request to Bedrock model {model_name}",
                details={'error_code': error_code, 'model': model_name},
                correlation_id=correlation_id
            )
        else:
            raise ExternalServiceError(
                f"Bedrock service error: {error}",
                service="bedrock",
                details={'error_code': error_code, 'model': model_name},
                correlation_id=correlation_id
            )
    else:
        raise ExternalServiceError(
            f"Unexpected Bedrock error: {error}",
            service="bedrock",
            details={'model': model_name},
            correlation_id=correlation_id
        )

def handle_s3_error(error: Exception, bucket: str, key: str = None, correlation_id: str = None) -> None:
    """Handle S3-specific errors"""
    if isinstance(error, ClientError):
        error_code = error.response.get('Error', {}).get('Code', 'Unknown')

        if error_code == 'NoSuchBucket':
            raise ExternalServiceError(
                f"S3 bucket '{bucket}' does not exist",
                service="s3",
                details={'bucket': bucket, 'key': key},
                correlation_id=correlation_id
            )
        elif error_code == 'NoSuchKey':
            raise ExternalServiceError(
                f"S3 object '{key}' not found in bucket '{bucket}'",
                service="s3",
                details={'bucket': bucket, 'key': key},
                correlation_id=correlation_id
            )
        elif error_code == 'AccessDenied':
            raise ExternalServiceError(
                f"Access denied to S3 resource",
                service="s3",
                details={'bucket': bucket, 'key': key},
                severity=ErrorSeverity.CRITICAL,
                correlation_id=correlation_id
            )
        else:
            raise ExternalServiceError(
                f"S3 service error: {error}",
                service="s3",
                details={'error_code': error_code, 'bucket': bucket, 'key': key},
                correlation_id=correlation_id
            )
    else:
        raise ExternalServiceError(
            f"Unexpected S3 error: {error}",
            service="s3",
            details={'bucket': bucket, 'key': key},
            correlation_id=correlation_id
        )