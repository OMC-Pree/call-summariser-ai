#!/usr/bin/env python3
"""
Retry mechanism with exponential backoff for external service calls

Provides resilient calling patterns for Bedrock, S3, and other AWS services.
"""

import time
from constants import MODEL_ID
import random
import logging
from typing import Callable, Any, Optional, List, Type
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import boto3
from botocore.exceptions import ClientError

class RetryStrategy(Enum):
    """Different retry strategies for different scenarios"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF

    # Error codes that should trigger retries
    retryable_errors: List[str] = None

    def __post_init__(self):
        if self.retryable_errors is None:
            self.retryable_errors = [
                'ThrottlingException',
                'ServiceUnavailableException',
                'InternalServerError',
                'RequestTimeout',
                'TooManyRequestsException',
                'ProvisionedThroughputExceededException'
            ]

class CircuitBreaker:
    """Circuit breaker pattern to prevent cascade failures"""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Reset circuit breaker on successful call"""
        self.failure_count = 0
        self.state = 'CLOSED'

    def _on_failure(self):
        """Handle failure in circuit breaker"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

class RetryHandler:
    """Main retry handler with multiple strategies"""

    def __init__(self, config: RetryConfig = None, logger: logging.Logger = None):
        self.config = config or RetryConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.circuit_breakers = {}

    def retry_call(self, func: Callable, *args, circuit_breaker_key: str = None,
                   custom_config: RetryConfig = None, **kwargs) -> Any:
        """Execute function with retry logic"""

        config = custom_config or self.config
        circuit_breaker = None

        # Use circuit breaker if key provided
        if circuit_breaker_key:
            if circuit_breaker_key not in self.circuit_breakers:
                self.circuit_breakers[circuit_breaker_key] = CircuitBreaker()
            circuit_breaker = self.circuit_breakers[circuit_breaker_key]

        last_exception = None

        for attempt in range(config.max_attempts):
            try:
                if circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if error is retryable
                if not self._is_retryable_error(e, config):
                    self.logger.info(f"Non-retryable error on attempt {attempt + 1}: {e}")
                    raise

                # Don't retry on last attempt
                if attempt == config.max_attempts - 1:
                    break

                # Calculate delay
                delay = self._calculate_delay(attempt, config)

                self.logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}",
                    extra={
                        'attempt': attempt + 1,
                        'max_attempts': config.max_attempts,
                        'delay': delay,
                        'error_type': type(e).__name__
                    }
                )

                time.sleep(delay)

        # All attempts failed
        self.logger.error(
            f"All {config.max_attempts} attempts failed",
            extra={'final_error': str(last_exception)}
        )
        raise last_exception

    def _is_retryable_error(self, error: Exception, config: RetryConfig) -> bool:
        """Determine if error should trigger a retry"""

        # Check for AWS ClientError with retryable error codes
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')
            return error_code in config.retryable_errors

        # Check for common transient errors
        transient_errors = [
            ConnectionError,
            TimeoutError,
            OSError  # Network errors
        ]

        return any(isinstance(error, err_type) for err_type in transient_errors)

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay based on retry strategy"""

        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.exponential_base ** attempt)
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * (attempt + 1)
        else:  # FIXED_DELAY
            delay = config.base_delay

        # Apply maximum delay limit
        delay = min(delay, config.max_delay)

        # Add jitter to prevent thundering herd
        if config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.1, delay)  # Minimum 100ms delay

def with_retry(config: RetryConfig = None, circuit_breaker_key: str = None):
    """Decorator for adding retry logic to functions"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_handler = RetryHandler(config)
            return retry_handler.retry_call(
                func, *args,
                circuit_breaker_key=circuit_breaker_key,
                **kwargs
            )
        return wrapper
    return decorator

# Pre-configured retry decorators for common services
bedrock_retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_errors=[
        'ThrottlingException',
        'ServiceUnavailableException',
        'InternalServerError',
        'ModelTimeoutException'
    ]
)

s3_retry_config = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    max_delay=10.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_errors=[
        'InternalError',
        'ServiceUnavailable',
        'SlowDown',
        'RequestTimeout'
    ]
)

dynamodb_retry_config = RetryConfig(
    max_attempts=5,
    base_delay=0.1,
    max_delay=5.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_errors=[
        'ProvisionedThroughputExceededException',
        'ThrottlingException',
        'InternalServerError',
        'ServiceUnavailableException'
    ]
)

# Convenient decorators
def with_bedrock_retry(circuit_breaker_key: str = "bedrock"):
    """Retry decorator optimized for Bedrock calls"""
    return with_retry(bedrock_retry_config, circuit_breaker_key)

def with_s3_retry(circuit_breaker_key: str = "s3"):
    """Retry decorator optimized for S3 calls"""
    return with_retry(s3_retry_config, circuit_breaker_key)

def with_dynamodb_retry(circuit_breaker_key: str = "dynamodb"):
    """Retry decorator optimized for DynamoDB calls"""
    return with_retry(dynamodb_retry_config, circuit_breaker_key)

# Utility functions for specific AWS service calls
class BedrockRetryWrapper:
    """Wrapper for Bedrock calls with built-in retry logic"""

    def __init__(self, client=None):
        self.client = client or boto3.client('bedrock-runtime')
        self.retry_handler = RetryHandler(bedrock_retry_config)

    @with_bedrock_retry()
    def invoke_model(self, model_id: str, body: dict) -> dict:
        """Invoke Bedrock model with retry logic"""
        import json

        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )

        return json.loads(response['body'].read())

class S3RetryWrapper:
    """Wrapper for S3 calls with built-in retry logic"""

    def __init__(self, client=None):
        self.client = client or boto3.client('s3')
        self.retry_handler = RetryHandler(s3_retry_config)

    @with_s3_retry()
    def put_object(self, bucket: str, key: str, body: Any, **kwargs) -> dict:
        """Put S3 object with retry logic"""
        return self.client.put_object(Bucket=bucket, Key=key, Body=body, **kwargs)

    @with_s3_retry()
    def get_object(self, bucket: str, key: str, **kwargs) -> dict:
        """Get S3 object with retry logic"""
        return self.client.get_object(Bucket=bucket, Key=key, **kwargs)

class DynamoDBRetryWrapper:
    """Wrapper for DynamoDB calls with built-in retry logic"""

    def __init__(self, table_name: str, client=None):
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb') if client is None else client
        self.table = self.dynamodb.Table(table_name)
        self.retry_handler = RetryHandler(dynamodb_retry_config)

    @with_dynamodb_retry()
    def put_item(self, item: dict = None, Item: dict = None, **kwargs) -> dict:
        """Put DynamoDB item with retry logic"""
        if item is None and Item is not None:
            item = Item
        if item is None:
            raise ValueError("put_item requires `item` (or `Item`).")
        return self.table.put_item(Item=item, **kwargs)

    @with_dynamodb_retry()
    def get_item(self, key: dict = None, Key: dict = None, **kwargs) -> dict:
        """Get DynamoDB item with retry logic"""
        if key is None and Key is not None:
            key = Key
        if key is None:
            raise ValueError("get_item requires `key` (or `Key`).")
        return self.table.get_item(Key=key, **kwargs)

    @with_dynamodb_retry()
    def update_item(self, key: dict = None, Key: dict = None, **kwargs) -> dict:
        """Update DynamoDB item with retry logic"""
        if key is None and Key is not None:
            key = Key
        if key is None:
            raise ValueError("update_item requires `key` (or `Key`).")
        return self.table.update_item(Key=key, **kwargs)

# Example usage in Lambda functions
def example_bedrock_call():
    """Example of using Bedrock with retry logic"""
    bedrock_wrapper = BedrockRetryWrapper()

    try:
        response = bedrock_wrapper.invoke_model(
            model_id=MODEL_ID,
            body={
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        return response
    except Exception as e:
        logging.error(f"Bedrock call failed after retries: {e}")
        raise

def example_s3_call():
    """Example of using S3 with retry logic"""
    s3_wrapper = S3RetryWrapper()

    try:
        response = s3_wrapper.put_object(
            bucket="my-bucket",
            key="my-key",
            body="my content",
            ContentType="text/plain"
        )
        return response
    except Exception as e:
        logging.error(f"S3 call failed after retries: {e}")
        raise