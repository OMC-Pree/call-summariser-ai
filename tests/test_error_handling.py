#!/usr/bin/env python3
"""
Unit tests for error handling and validation components

Tests the error handling framework, input validation, and retry mechanisms.
"""

import unittest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

# Add path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'summariser'))

from utils.error_handler import (
    ErrorHandler, CallSummarizerError, ValidationError, ExternalServiceError,
    BusinessLogicError, InputValidator, lambda_error_handler,
    ErrorCategory, ErrorSeverity
)
from utils.retry_handler import (
    RetryHandler, RetryConfig, RetryStrategy, CircuitBreaker,
    with_retry, BedrockRetryWrapper, S3RetryWrapper, DynamoDBRetryWrapper
)

class TestErrorHandler(unittest.TestCase):
    """Test error handling framework"""

    def setUp(self):
        self.error_handler = ErrorHandler()

    def test_handle_validation_error(self):
        """Test handling of validation errors"""
        error = ValidationError("Invalid field", field="testField")
        context = {'correlation_id': 'test-123'}

        result = self.error_handler.handle_error(error, context)

        self.assertIn('error', result)
        self.assertEqual(result['error']['category'], 'USER_INPUT')
        self.assertEqual(result['error']['correlation_id'], 'test-123')

    def test_handle_external_service_error(self):
        """Test handling of external service errors"""
        error = ExternalServiceError("Service unavailable", service="bedrock")
        result = self.error_handler.handle_error(error)

        self.assertIn('error', result)
        self.assertEqual(result['error']['category'], 'EXTERNAL_SERVICE')

    def test_handle_unknown_error(self):
        """Test handling of unknown/unexpected errors"""
        error = ValueError("Something went wrong")
        result = self.error_handler.handle_error(error)

        self.assertIn('error', result)
        self.assertIn('error_id', result['error'])

    def test_categorize_client_error(self):
        """Test categorization of AWS ClientError"""
        client_error = ClientError(
            error_response={'Error': {'Code': 'ThrottlingException'}},
            operation_name='test'
        )

        category, severity = self.error_handler._categorize_error(client_error)
        self.assertEqual(category, ErrorCategory.EXTERNAL_SERVICE)
        self.assertEqual(severity, ErrorSeverity.HIGH)

    @patch('boto3.client')
    def test_send_cloudwatch_metric(self, mock_boto3):
        """Test CloudWatch metric sending"""
        mock_cloudwatch = Mock()
        mock_boto3.return_value = mock_cloudwatch

        error_handler = ErrorHandler()
        error_handler._send_error_metric('TEST_CATEGORY', 'HIGH')

        mock_cloudwatch.put_metric_data.assert_called_once()

class TestInputValidator(unittest.TestCase):
    """Test input validation utilities"""

    def test_validate_required_fields_success(self):
        """Test successful validation of required fields"""
        data = {'field1': 'value1', 'field2': 'value2'}
        required = ['field1', 'field2']

        # Should not raise an exception
        InputValidator.validate_required_fields(data, required)

    def test_validate_required_fields_missing(self):
        """Test validation failure with missing fields"""
        data = {'field1': 'value1'}
        required = ['field1', 'field2']

        with self.assertRaises(ValidationError) as context:
            InputValidator.validate_required_fields(data, required)

        self.assertIn('field2', str(context.exception))

    def test_validate_string_field_success(self):
        """Test successful string field validation"""
        result = InputValidator.validate_string_field("test value", "testField")
        self.assertEqual(result, "test value")

    def test_validate_string_field_too_short(self):
        """Test string field validation with too short value"""
        with self.assertRaises(ValidationError):
            InputValidator.validate_string_field("", "testField", min_length=1)

    def test_validate_string_field_too_long(self):
        """Test string field validation with too long value"""
        with self.assertRaises(ValidationError):
            InputValidator.validate_string_field("x" * 1000, "testField", max_length=100)

    def test_validate_string_field_not_string(self):
        """Test string field validation with non-string input"""
        with self.assertRaises(ValidationError):
            InputValidator.validate_string_field(123, "testField")

    def test_validate_meeting_id_success(self):
        """Test successful meeting ID validation"""
        result = InputValidator.validate_meeting_id("12345678901")
        self.assertEqual(result, "12345678901")

    def test_validate_meeting_id_invalid_format(self):
        """Test meeting ID validation with invalid format"""
        invalid_ids = ["abc123", "123", "12345678901234567890123", ""]

        for invalid_id in invalid_ids:
            with self.assertRaises(ValidationError):
                InputValidator.validate_meeting_id(invalid_id)

    def test_sanitize_text_removes_scripts(self):
        """Test text sanitization removes script tags"""
        malicious_text = "Hello <script>alert('xss')</script> world"
        result = InputValidator.sanitize_text(malicious_text)
        self.assertNotIn('<script>', result)
        self.assertIn('Hello', result)
        self.assertIn('world', result)

    def test_sanitize_text_length_limit(self):
        """Test text sanitization enforces length limits"""
        long_text = "x" * 200000  # 200KB
        result = InputValidator.sanitize_text(long_text)
        self.assertLessEqual(len(result), 100000)

class TestLambdaErrorHandler(unittest.TestCase):
    """Test Lambda error handler decorator"""

    def test_lambda_error_handler_success(self):
        """Test successful Lambda function execution"""
        @lambda_error_handler()
        def test_function(event, context):
            return {'statusCode': 200, 'body': 'success'}

        context = Mock()
        context.aws_request_id = 'test-request-123'
        context.function_name = 'test-function'

        result = test_function({}, context)
        self.assertEqual(result['statusCode'], 200)

    def test_lambda_error_handler_validation_error(self):
        """Test Lambda error handler with validation error"""
        @lambda_error_handler()
        def test_function(event, context):
            raise ValidationError("Invalid input")

        context = Mock()
        context.aws_request_id = 'test-request-123'
        context.function_name = 'test-function'

        result = test_function({}, context)
        self.assertEqual(result['statusCode'], 400)
        self.assertIn('error', json.loads(result['body']))

    def test_lambda_error_handler_internal_error(self):
        """Test Lambda error handler with internal error"""
        @lambda_error_handler()
        def test_function(event, context):
            raise Exception("Internal error")

        context = Mock()
        context.aws_request_id = 'test-request-123'
        context.function_name = 'test-function'

        result = test_function({}, context)
        self.assertEqual(result['statusCode'], 500)

class TestRetryHandler(unittest.TestCase):
    """Test retry mechanism"""

    def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt"""
        retry_handler = RetryHandler()
        mock_function = Mock(return_value="success")

        result = retry_handler.retry_call(mock_function)

        self.assertEqual(result, "success")
        mock_function.assert_called_once()

    def test_retry_success_after_failures(self):
        """Test successful execution after retries"""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry_handler = RetryHandler(config)

        mock_function = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])

        result = retry_handler.retry_call(mock_function)

        self.assertEqual(result, "success")
        self.assertEqual(mock_function.call_count, 3)

    def test_retry_max_attempts_exceeded(self):
        """Test failure after max attempts"""
        config = RetryConfig(max_attempts=2, base_delay=0.1)
        retry_handler = RetryHandler(config)

        mock_function = Mock(side_effect=Exception("persistent failure"))

        with self.assertRaises(Exception):
            retry_handler.retry_call(mock_function)

        self.assertEqual(mock_function.call_count, 2)

    def test_retry_non_retryable_error(self):
        """Test immediate failure for non-retryable errors"""
        config = RetryConfig(max_attempts=3, retryable_errors=['ThrottlingException'])
        retry_handler = RetryHandler(config)

        # ValueError is not in retryable_errors list
        mock_function = Mock(side_effect=ValueError("validation error"))

        with self.assertRaises(ValueError):
            retry_handler.retry_call(mock_function)

        mock_function.assert_called_once()

    def test_retry_with_client_error(self):
        """Test retry with AWS ClientError"""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry_handler = RetryHandler(config)

        # Create a retryable ClientError
        client_error = ClientError(
            error_response={'Error': {'Code': 'ThrottlingException'}},
            operation_name='test'
        )

        mock_function = Mock(side_effect=[client_error, "success"])
        result = retry_handler.retry_call(mock_function)

        self.assertEqual(result, "success")
        self.assertEqual(mock_function.call_count, 2)

class TestRetryConfig(unittest.TestCase):
    """Test retry configuration"""

    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation"""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=False
        )
        retry_handler = RetryHandler(config)

        delay_0 = retry_handler._calculate_delay(0, config)
        delay_1 = retry_handler._calculate_delay(1, config)
        delay_2 = retry_handler._calculate_delay(2, config)

        self.assertEqual(delay_0, 1.0)
        self.assertEqual(delay_1, 2.0)
        self.assertEqual(delay_2, 4.0)

    def test_linear_backoff_calculation(self):
        """Test linear backoff delay calculation"""
        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            jitter=False
        )
        retry_handler = RetryHandler(config)

        delay_0 = retry_handler._calculate_delay(0, config)
        delay_1 = retry_handler._calculate_delay(1, config)
        delay_2 = retry_handler._calculate_delay(2, config)

        self.assertEqual(delay_0, 1.0)
        self.assertEqual(delay_1, 2.0)
        self.assertEqual(delay_2, 3.0)

    def test_max_delay_limit(self):
        """Test maximum delay limit"""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=10.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=False
        )
        retry_handler = RetryHandler(config)

        delay = retry_handler._calculate_delay(5, config)  # Would be 100s without limit
        self.assertEqual(delay, 5.0)

class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker pattern"""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        cb = CircuitBreaker(failure_threshold=3, timeout=1.0)
        mock_function = Mock(return_value="success")

        result = cb.call(mock_function)

        self.assertEqual(result, "success")
        self.assertEqual(cb.state, 'CLOSED')

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after failure threshold"""
        cb = CircuitBreaker(failure_threshold=2, timeout=1.0)
        mock_function = Mock(side_effect=Exception("failure"))

        # First failure
        with self.assertRaises(Exception):
            cb.call(mock_function)
        self.assertEqual(cb.state, 'CLOSED')

        # Second failure - should open circuit
        with self.assertRaises(Exception):
            cb.call(mock_function)
        self.assertEqual(cb.state, 'OPEN')

        # Third call should fail immediately without calling function
        with self.assertRaises(Exception) as context:
            cb.call(mock_function)

        self.assertIn("Circuit breaker is OPEN", str(context.exception))
        self.assertEqual(mock_function.call_count, 2)  # Not called the third time

class TestRetryDecorators(unittest.TestCase):
    """Test retry decorators"""

    def test_with_retry_decorator(self):
        """Test with_retry decorator"""
        config = RetryConfig(max_attempts=2, base_delay=0.1)

        @with_retry(config)
        def test_function():
            if not hasattr(test_function, 'call_count'):
                test_function.call_count = 0
            test_function.call_count += 1
            if test_function.call_count < 2:
                raise Exception("retry me")
            return "success"

        result = test_function()
        self.assertEqual(result, "success")
        self.assertEqual(test_function.call_count, 2)

class TestServiceWrappers(unittest.TestCase):
    """Test service-specific retry wrappers"""

    @patch('boto3.client')
    def test_bedrock_retry_wrapper(self, mock_boto3):
        """Test Bedrock retry wrapper"""
        mock_client = Mock()
        mock_boto3.return_value = mock_client

        mock_response = {
            'body': Mock(read=Mock(return_value=json.dumps({
                'content': [{'text': 'Hello'}]
            }).encode()))
        }
        mock_client.invoke_model.return_value = mock_response

        wrapper = BedrockRetryWrapper(mock_client)
        result = wrapper.invoke_model("test-model", {"test": "data"})

        self.assertIn('content', result)
        mock_client.invoke_model.assert_called_once()

    @patch('boto3.client')
    def test_s3_retry_wrapper(self, mock_boto3):
        """Test S3 retry wrapper"""
        mock_client = Mock()
        mock_boto3.return_value = mock_client

        mock_client.put_object.return_value = {'ETag': 'test-etag'}

        wrapper = S3RetryWrapper(mock_client)
        result = wrapper.put_object("test-bucket", "test-key", "test-body")

        self.assertEqual(result['ETag'], 'test-etag')
        mock_client.put_object.assert_called_once()

    @patch('boto3.resource')
    def test_dynamodb_retry_wrapper(self, mock_boto3):
        """Test DynamoDB retry wrapper"""
        mock_resource = Mock()
        mock_table = Mock()
        mock_resource.Table.return_value = mock_table
        mock_boto3.return_value = mock_resource

        mock_table.put_item.return_value = {'ResponseMetadata': {'HTTPStatusCode': 200}}

        wrapper = DynamoDBRetryWrapper("test-table", mock_resource)
        result = wrapper.put_item({"test": "data"})

        self.assertEqual(result['ResponseMetadata']['HTTPStatusCode'], 200)
        mock_table.put_item.assert_called_once()

if __name__ == '__main__':
    unittest.main()