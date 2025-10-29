"""
AWS Bedrock Prompt Management integration

Utilities for fetching prompts from Bedrock Prompt Management and invoking them with tools.
Architecture: Prompts in PM, tool definitions in code, combined at runtime.
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from utils import helper
from utils.aws_clients import AWSClients

bedrock_runtime = AWSClients.bedrock_runtime()
bedrock_agent = None


def _get_bedrock_agent():
    """Get Bedrock Agent client (lazily initialized)"""
    global bedrock_agent
    if bedrock_agent is None:
        import boto3
        from constants import AWS_REGION
        bedrock_agent = boto3.client("bedrock-agent", region_name=AWS_REGION)
    return bedrock_agent


def get_prompt_arn_from_parameter_store(
    param_name: str,
    cache_dict: Dict[str, Optional[str]],
    use_prompt_management: bool = True
) -> Optional[str]:
    """
    Get prompt ARN from Parameter Store with Lambda container caching.

    Args:
        param_name: SSM parameter name
        cache_dict: Cache dictionary (module-level)
        use_prompt_management: Whether prompt management is enabled

    Returns:
        Prompt ARN or None
    """
    cache_key = "arn"

    if not use_prompt_management:
        return None

    if cache_key in cache_dict and cache_dict[cache_key] is not None:
        return cache_dict[cache_key]

    try:
        ssm = AWSClients.ssm()
        response = ssm.get_parameter(Name=param_name)
        cache_dict[cache_key] = response['Parameter']['Value']
        helper.log_json("INFO", "PROMPT_ARN_LOADED",
                       parameter_name=param_name,
                       prompt_arn=cache_dict[cache_key])
        return cache_dict[cache_key]
    except Exception as e:
        helper.log_json("WARNING", "PROMPT_ARN_LOAD_FAILED",
                       parameter_name=param_name,
                       error=str(e))
        return None


def _parse_prompt_arn(prompt_arn: str) -> Tuple[str, Optional[str]]:
    """Parse prompt ARN to extract ID and version."""
    last_segment = prompt_arn.split("/")[-1]

    if ":" in last_segment:
        parts = last_segment.split(":")
        prompt_id = parts[0]
        version = parts[1]
    else:
        prompt_id = last_segment
        version = None

    return prompt_id, version


def get_prompt_text(prompt_arn: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Fetch prompt text and inference config from Prompt Management.

    Returns: (system_prompt, user_template, inference_config)
    """
    prompt_id, version = _parse_prompt_arn(prompt_arn)

    try:
        agent = _get_bedrock_agent()
        if version:
            response = agent.get_prompt(
                promptIdentifier=prompt_id,
                promptVersion=version
            )
        else:
            response = agent.get_prompt(promptIdentifier=prompt_id)

        if not response.get('variants'):
            raise ValueError(f"No variants found for prompt {prompt_id} (version: {version})")

        variant = response['variants'][0]
        system_prompt = ""
        template_config = variant.get('templateConfiguration', {}).get('text', {})
        user_prompt_template = template_config.get('text', '')

        inference_config = {}
        if 'inferenceConfiguration' in variant:
            inf_conf = variant['inferenceConfiguration'].get('text', {})
            inference_config = {
                'temperature': inf_conf.get('temperature', 0.3),
                'maxTokens': inf_conf.get('maxTokens', 4000),
                'topP': inf_conf.get('topP', 1.0)
            }

        helper.log_json("INFO", "PROMPT_FETCHED_FROM_PM",
                       prompt_id=prompt_id,
                       version=version,
                       template_length=len(user_prompt_template))

        return system_prompt, user_prompt_template, inference_config

    except Exception as e:
        helper.log_json("ERROR", "PROMPT_FETCH_FAILED",
                       prompt_arn=prompt_arn,
                       error=str(e))
        raise


def invoke_with_prompt_management(
    prompt_arn: str,
    variables: Dict[str, str],
    model_id: str,
    tools: List[Dict[str, Any]],
    tool_choice: Optional[Dict[str, Any]] = None,
    system_override: Optional[str] = None,
    max_tokens_override: Optional[int] = None
) -> Tuple[Dict[str, Any], int]:
    """
    Fetch prompt from Prompt Management and invoke Converse API with tools.

    Returns: (response, latency_ms)
    """
    system_pm, user_template_pm, inference_config = get_prompt_text(prompt_arn)

    user_prompt = user_template_pm
    for key, value in variables.items():
        user_prompt = user_prompt.replace(f"{{{{{key}}}}}", value)

    messages = [{"role": "user", "content": [{"text": user_prompt}]}]

    request_params = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {
            "temperature": inference_config.get('temperature', 0.3),
            "maxTokens": max_tokens_override or inference_config.get('maxTokens', 4000),
            "topP": inference_config.get('topP', 1.0)
        }
    }

    system_text = system_override or system_pm
    if system_text:
        request_params["system"] = [{"text": system_text}]

    request_params["toolConfig"] = {"tools": tools}
    if tool_choice:
        request_params["toolConfig"]["toolChoice"] = tool_choice
    elif len(tools) == 1:
        tool_name = tools[0].get("toolSpec", {}).get("name")
        if tool_name:
            request_params["toolConfig"]["toolChoice"] = {"tool": {"name": tool_name}}
        else:
            helper.log_json("WARNING", "MISSING_TOOL_NAME", tool_structure=str(tools[0]))

    start_time = time.time()

    try:
        response = bedrock_runtime.converse(**request_params)
        latency_ms = int((time.time() - start_time) * 1000)

        return response, latency_ms

    except Exception as e:
        helper.log_json("ERROR", "CONVERSE_WITH_PM_FAILED",
                       prompt_arn=prompt_arn,
                       error=str(e))
        raise


def get_prompt_info(prompt_arn: str) -> Dict[str, Any]:
    """Get prompt metadata from Prompt Management."""
    prompt_id, version = _parse_prompt_arn(prompt_arn)

    agent = _get_bedrock_agent()
    if version:
        response = agent.get_prompt(
            promptIdentifier=prompt_id,
            promptVersion=version
        )
    else:
        response = agent.get_prompt(promptIdentifier=prompt_id)

    return {
        "id": response["id"],
        "name": response["name"],
        "description": response.get("description", ""),
        "version": response.get("version", "DRAFT"),
        "created_at": response.get("createdAt"),
        "updated_at": response.get("updatedAt")
    }
