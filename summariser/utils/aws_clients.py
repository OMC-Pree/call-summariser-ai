"""
AWS Client Management - Centralized boto3 client creation

This module provides singleton boto3 clients to avoid duplicate initialization
across Lambda functions. Clients are created once per Lambda container lifecycle.
"""
import boto3
from typing import Optional
from constants import AWS_REGION


class AWSClients:
    """Singleton manager for AWS service clients"""

    _bedrock_runtime: Optional[object] = None
    _bedrock_agent_runtime: Optional[object] = None
    _s3: Optional[object] = None
    _ssm: Optional[object] = None
    _comprehend: Optional[object] = None
    _stepfunctions: Optional[object] = None
    _a2i: Optional[object] = None
    _dynamodb: Optional[object] = None

    @classmethod
    def bedrock_runtime(cls):
        """Get Bedrock Runtime client for model invocations"""
        if cls._bedrock_runtime is None:
            cls._bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        return cls._bedrock_runtime

    @classmethod
    def bedrock_agent_runtime(cls):
        """Get Bedrock Agent Runtime client for Knowledge Base operations"""
        if cls._bedrock_agent_runtime is None:
            cls._bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
        return cls._bedrock_agent_runtime

    @classmethod
    def s3(cls):
        """Get S3 client for object storage operations"""
        if cls._s3 is None:
            cls._s3 = boto3.client("s3", region_name=AWS_REGION)
        return cls._s3

    @classmethod
    def ssm(cls):
        """Get Systems Manager client for Parameter Store operations"""
        if cls._ssm is None:
            cls._ssm = boto3.client("ssm", region_name=AWS_REGION)
        return cls._ssm

    @classmethod
    def comprehend(cls):
        """Get Comprehend client for PII detection"""
        if cls._comprehend is None:
            cls._comprehend = boto3.client("comprehend", region_name=AWS_REGION)
        return cls._comprehend

    @classmethod
    def stepfunctions(cls):
        """Get Step Functions client for workflow orchestration"""
        if cls._stepfunctions is None:
            cls._stepfunctions = boto3.client("stepfunctions", region_name=AWS_REGION)
        return cls._stepfunctions

    @classmethod
    def a2i(cls):
        """Get SageMaker A2I Runtime client for human review workflows"""
        if cls._a2i is None:
            cls._a2i = boto3.client("sagemaker-a2i-runtime", region_name=AWS_REGION)
        return cls._a2i

    @classmethod
    def dynamodb(cls):
        """Get DynamoDB client for database operations"""
        if cls._dynamodb is None:
            cls._dynamodb = boto3.client("dynamodb", region_name=AWS_REGION)
        return cls._dynamodb


# Convenience exports for backward compatibility
def get_bedrock_client():
    """Get Bedrock Runtime client"""
    return AWSClients.bedrock_runtime()


def get_s3_client():
    """Get S3 client"""
    return AWSClients.s3()


def get_ssm_client():
    """Get SSM client"""
    return AWSClients.ssm()
