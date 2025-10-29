#!/usr/bin/env python3
"""
Script to create and manage prompts in AWS Bedrock Prompt Management

This script automates the creation of prompts in Bedrock Prompt Management
using the AWS SDK, so you don't have to manually configure them in the console.

Usage:
    python create_prompts.py --create-all
    python create_prompts.py --create summary
    python create_prompts.py --create case-check
    python create_prompts.py --list
"""

import json
import boto3
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

# Initialize Bedrock Agent client (used for Prompt Management)
bedrock_agent = boto3.client('bedrock-agent', region_name='eu-west-2')


def load_prompt_template(template_path: str) -> Dict[str, Any]:
    """Load prompt template from JSON file"""
    with open(template_path, 'r') as f:
        return json.load(f)


def create_prompt(template_data: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Create a prompt in Bedrock Prompt Management

    Returns:
        Tuple of (prompt_id, prompt_arn, version, version_arn)
    """
    print(f"Creating prompt: {template_data['name']}")

    # Validate required fields
    required_fields = ['name', 'userPrompt', 'modelId', 'inferenceConfiguration']
    missing_fields = [field for field in required_fields if field not in template_data]
    if missing_fields:
        raise ValueError(f"Missing required fields in template: {missing_fields}")

    # Validate userPrompt structure
    if not template_data.get('userPrompt') or not isinstance(template_data['userPrompt'], list) or len(template_data['userPrompt']) == 0:
        raise ValueError("userPrompt must be a non-empty list")
    if 'text' not in template_data['userPrompt'][0]:
        raise ValueError("userPrompt must contain at least one entry with 'text' field")

    # Build the full prompt text including system message
    # Bedrock Prompt Management doesn't directly support system prompts,
    # so we'll include instructions in the user prompt
    full_prompt = template_data['userPrompt'][0]['text']

    if 'systemPrompt' in template_data and template_data['systemPrompt']:
        if not isinstance(template_data['systemPrompt'], list) or len(template_data['systemPrompt']) == 0:
            raise ValueError("systemPrompt must be a non-empty list if provided")
        if 'text' not in template_data['systemPrompt'][0]:
            raise ValueError("systemPrompt must contain at least one entry with 'text' field")
        system_text = template_data['systemPrompt'][0]['text']
        # Prepend system instructions to user prompt
        full_prompt = f"{system_text}\n\n{full_prompt}"

    # Prepare the variant configuration
    variant = {
        'name': 'default',
        'templateType': 'TEXT',
        'modelId': template_data['modelId'],
        'inferenceConfiguration': {
            'text': {
                'temperature': template_data['inferenceConfiguration']['temperature'],
                'maxTokens': template_data['inferenceConfiguration']['maxTokens'],
                'topP': template_data['inferenceConfiguration'].get('topP', 1.0)
            }
        },
        'templateConfiguration': {
            'text': {
                'text': full_prompt,
                'inputVariables': [
                    {'name': var['name']} for var in template_data.get('variables', [])
                ]
            }
        }
    }

    # Prepare the request
    request_params = {
        'name': template_data['name'],
        'description': template_data.get('description', ''),
        'variants': [variant]
    }

    # Note: Tool configurations are kept in the Lambda code
    # They will be passed at runtime when invoking via Converse API
    # Prompt Management stores only the prompt text and inference config

    prompt_id = None
    try:
        response = bedrock_agent.create_prompt(**request_params)
        prompt_id = response['id']
        prompt_arn = response['arn']

        print(f"✅ Prompt created successfully!")
        print(f"   ID: {prompt_id}")
        print(f"   ARN: {prompt_arn}")

        # Create a version of the prompt
        print(f"Creating version for prompt: {prompt_id}")
        try:
            version_response = bedrock_agent.create_prompt_version(
                promptIdentifier=prompt_id,
                description="Initial version created by automation script"
            )

            version = version_response['version']
            version_arn = version_response['arn']

            print(f"✅ Version created successfully!")
            print(f"   Version: {version}")
            print(f"   Version ARN: {version_arn}")

            return prompt_id, prompt_arn, version, version_arn

        except Exception as version_error:
            # Version creation failed, clean up the prompt
            print(f"⚠️  Version creation failed: {version_error}")
            print(f"   Attempting to delete orphaned prompt {prompt_id}...")
            try:
                bedrock_agent.delete_prompt(promptIdentifier=prompt_id)
                print(f"   ✅ Cleaned up prompt {prompt_id}")
            except Exception as cleanup_error:
                print(f"   ⚠️  Failed to cleanup prompt {prompt_id}: {cleanup_error}")
            raise version_error

    except Exception as e:
        if prompt_id:
            print(f"❌ Error after prompt creation: {e}")
        else:
            print(f"❌ Error creating prompt: {e}")
        raise


def list_prompts():
    """List all prompts in Prompt Management"""
    try:
        response = bedrock_agent.list_prompts()
        prompts = response.get('promptSummaries', [])

        if not prompts:
            print("No prompts found")
            return

        print(f"\n📋 Found {len(prompts)} prompts:")
        print("-" * 80)

        for prompt in prompts:
            print(f"Name: {prompt['name']}")
            print(f"ID: {prompt['id']}")
            print(f"ARN: {prompt['arn']}")
            print(f"Created: {prompt['createdAt']}")
            print(f"Updated: {prompt['updatedAt']}")
            print("-" * 80)

    except Exception as e:
        print(f"❌ Error listing prompts: {e}")


def get_prompt_details(prompt_id: str):
    """Get detailed information about a prompt"""
    try:
        response = bedrock_agent.get_prompt(promptIdentifier=prompt_id)
        print(json.dumps(response, indent=2, default=str))
    except Exception as e:
        print(f"❌ Error getting prompt details: {e}")


def main():
    parser = argparse.ArgumentParser(description='Manage Bedrock Prompt Management prompts')
    parser.add_argument('--create', choices=['summary', 'case-check', 'all'],
                       help='Create prompt(s)')
    parser.add_argument('--list', action='store_true',
                       help='List all prompts')
    parser.add_argument('--get', type=str,
                       help='Get details for a specific prompt ID')

    args = parser.parse_args()

    script_dir = Path(__file__).parent

    if args.list:
        list_prompts()
        return

    if args.get:
        get_prompt_details(args.get)
        return

    if args.create:
        results = {}

        if args.create in ['summary', 'all']:
            print("\n" + "="*80)
            print("Creating Summary Prompt")
            print("="*80)

            template_path = script_dir / 'summary_prompt_template.json'
            template_data = load_prompt_template(template_path)

            try:
                prompt_id, prompt_arn, version, version_arn = create_prompt(template_data)
                results['summary'] = {
                    'id': prompt_id,
                    'arn': prompt_arn,
                    'version': version,
                    'version_arn': version_arn
                }
            except Exception as e:
                print(f"Failed to create summary prompt: {e}")

        if args.create in ['case-check', 'all']:
            print("\n" + "="*80)
            print("Creating Case Check Prompt")
            print("="*80)

            template_path = script_dir / 'case_check_prompt_template.json'
            template_data = load_prompt_template(template_path)

            try:
                prompt_id, prompt_arn, version, version_arn = create_prompt(template_data)
                results['case_check'] = {
                    'id': prompt_id,
                    'arn': prompt_arn,
                    'version': version,
                    'version_arn': version_arn
                }
            except Exception as e:
                print(f"Failed to create case check prompt: {e}")

        # Save results to file
        if results:
            output_file = script_dir / 'prompt_arns.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print("\n" + "="*80)
            print("✅ All prompts created successfully!")
            print("="*80)
            print(f"\n📁 ARNs saved to: {output_file}")
            print("\n💡 Next steps:")
            print("   1. Add these ARNs to your Lambda environment variables")
            print("   2. Update your Lambda code to use Prompt Management")
            print("   3. Run evals to test the prompts")

            print("\n📋 Environment Variables to add:")
            for key, value in results.items():
                env_var_name = f"PROMPT_ARN_{key.upper()}"
                print(f"   {env_var_name}={value['version_arn']}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
