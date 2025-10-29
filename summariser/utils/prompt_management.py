"""
Bedrock Prompt Management integration helper

This module provides utilities for working with prompts stored in AWS Bedrock Prompt Management.
It fetches prompt text and combines it with tools at runtime for flexible structured output.

ARCHITECTURE:
- Prompts stored in Prompt Management (easy to change via AWS Console)
- Tool definitions in Lambda code (co-located with Pydantic models)
- At runtime: fetch prompt + add tools + call Converse API
"""

import os
import boto3
from typing import Dict, Any, Optional, List, Tuple
from utils import helper

AWS_REGION = os.getenv("AWS_REGION", "eu-west-2")
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
bedrock_agent = boto3.client("bedrock-agent", region_name=AWS_REGION)


def _parse_prompt_arn(prompt_arn: str) -> Tuple[str, Optional[str]]:
    """
    Parse prompt ARN to extract prompt ID and version.

    Args:
        prompt_arn: ARN like arn:aws:bedrock:region:account:prompt/prompt-id:version
                    or arn:aws:bedrock:region:account:prompt/prompt-id (no version)

    Returns:
        tuple: (prompt_id, version or None)
    """
    # Extract the last segment after the final slash
    last_segment = prompt_arn.split("/")[-1]

    # Check if version exists (last segment contains a colon)
    if ":" in last_segment:
        # Split on colon to separate ID and version
        parts = last_segment.split(":")
        prompt_id = parts[0]
        version = parts[1]
    else:
        # No version, the entire last segment is the prompt ID
        prompt_id = last_segment
        version = None

    return prompt_id, version


def get_prompt_text(prompt_arn: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Fetch prompt text and configuration from Prompt Management.

    Args:
        prompt_arn: ARN of the prompt (including version) e.g. arn:aws:bedrock:...:prompt/ID:VERSION

    Returns:
        tuple: (system_prompt, user_prompt_template, inference_config)

    Example:
        system, user_template, config = get_prompt_text(prompt_arn)
        user_prompt = user_template.replace("{{transcript}}", transcript_text)
    """
    # Extract prompt ID and version from ARN
    prompt_id, version = _parse_prompt_arn(prompt_arn)

    # Get prompt details
    try:
        if version:
            response = bedrock_agent.get_prompt(
                promptIdentifier=prompt_id,
                promptVersion=version
            )
        else:
            response = bedrock_agent.get_prompt(promptIdentifier=prompt_id)

        # Extract the variant (we use 'default' variant)
        if not response.get('variants'):
            raise ValueError(f"No variants found for prompt {prompt_id} (version: {version})")

        variant = response['variants'][0]

        # Get system prompt (placeholder for now)
        system_prompt = ""

        # Get user prompt template
        template_config = variant.get('templateConfiguration', {}).get('text', {})
        user_prompt_template = template_config.get('text', '')

        # Get inference configuration
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
    Fetch prompt from Prompt Management and invoke with tools.

    This is the recommended approach: prompts in PM, tools in code.

    Args:
        prompt_arn: ARN of the prompt from Prompt Management
        variables: Variables to substitute in prompt template
        model_id: Model ID to use (e.g. "anthropic.claude-3-5-sonnet-20241022-v2:0")
        tools: Tool definitions for structured output
        tool_choice: Optional tool choice configuration
        system_override: Optional system prompt override
        max_tokens_override: Optional override for max_tokens (overrides prompt's inference config)

    Returns:
        tuple: (response dict, latency_ms)

    Example:
        response, latency = invoke_with_prompt_management(
            prompt_arn="arn:aws:bedrock:eu-west-2:123:prompt/ABC:1",
            variables={"transcript": transcript_text},
            model_id=MODEL_ID,
            tools=[get_summary_tool()],
            tool_choice={"tool": {"name": "submit_call_summary"}}
        )
    """
    import time

    # Fetch prompt text from Prompt Management
    system_pm, user_template_pm, inference_config = get_prompt_text(prompt_arn)

    # Substitute variables in template
    # Replace {{variable_name}} with actual values
    user_prompt = user_template_pm
    for key, value in variables.items():
        user_prompt = user_prompt.replace(f"{{{{{key}}}}}", value)

    # Build Converse API request
    messages = [
        {"role": "user", "content": [{"text": user_prompt}]}
    ]

    request_params = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {
            "temperature": inference_config.get('temperature', 0.3),
            "maxTokens": max_tokens_override or inference_config.get('maxTokens', 4000),
            "topP": inference_config.get('topP', 1.0)
        }
    }

    # Add system prompt
    system_text = system_override or system_pm
    if system_text:
        request_params["system"] = [{"text": system_text}]

    # Add tools
    request_params["toolConfig"] = {"tools": tools}
    if tool_choice:
        request_params["toolConfig"]["toolChoice"] = tool_choice
    elif len(tools) == 1:
        # Auto-force single tool use
        tool_name = tools[0].get("toolSpec", {}).get("name")
        if tool_name:
            request_params["toolConfig"]["toolChoice"] = {
                "tool": {"name": tool_name}
            }
        else:
            helper.log_json("WARNING", "MISSING_TOOL_NAME",
                          tool_structure=str(tools[0]))

    start_time = time.time()

    # Call Converse API
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
    """
    Get information about a prompt from Prompt Management.

    Args:
        prompt_arn: ARN of the prompt

    Returns:
        Dict with prompt metadata
    """
    # Extract prompt ID and version from ARN using helper
    prompt_id, version = _parse_prompt_arn(prompt_arn)

    response = bedrock_agent.get_prompt(promptIdentifier=prompt_id)

    return {
        "id": response["id"],
        "name": response["name"],
        "description": response.get("description", ""),
        "version": response.get("version", "DRAFT"),
        "created_at": response.get("createdAt"),
        "updated_at": response.get("updatedAt")
    }
