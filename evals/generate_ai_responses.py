"""
Script to process meetings through multiple AI models and store their
vulnerability assessments in DynamoDB.

This script:
1. Fetches meetings from DynamoDB (vulnerability-assessments table)
2. Processes each meeting through GPT-4o, Gemini Pro, and Claude 3.5
3. Stores AI responses with reasoning in the same DynamoDB record

Usage:
    python3 generate_ai_responses.py [--meeting-id MEETING_ID] [--limit N]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any
from pathlib import Path

import boto3
from boto3.dynamodb.conditions import Key

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from the evals directory
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
    print("⚠ Falling back to environment variables only")


# Initialize AWS clients
aws_region = os.environ.get('AWS_REGION', 'eu-west-2')
dynamodb = boto3.resource('dynamodb', region_name=aws_region)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)

# Get table names from environment
assessments_table_name = os.environ.get('ASSESSMENTS_TABLE', 'vulnerability-assessments')
assessments_table = dynamodb.Table(assessments_table_name)


VULNERABILITY_ASSESSMENT_PROMPT = """You are an expert vulnerability assessor. Analyze the following financial coaching conversation and assess the customer's vulnerability level.

VULNERABILITY RATING SCALE:
- Critical/5: Severe, multiple critical vulnerabilities (e.g., domestic violence, suicide risk, severe mental health crisis)
- High/4: Significant vulnerabilities requiring attention (e.g., chronic illness, learning difficulties, bereavement)
- Medium/3: Moderate vulnerabilities (e.g., temporary illness, low emotional resilience, situational anxiety)
- Low/2: Minor or potential vulnerabilities (e.g., low financial confidence, minor health issues)
- Marginal/1: Very minimal vulnerability indicators
- None/0: No vulnerability indicators

VULNERABILITY CATEGORIES:
- Capability: Learning Difficulties, Low Confidence Financial Matters, Low Mental Capacity, Young Or Old Age
- Health: Addiction, Chronic Illness, Mental Health Condition, Physical Disability, Suicide, Temporary Illness Or Hospitalisation Or Accident
- Life Events: Bereavement, Caring Responsibilities, Divorce Or Separation, Domestic Violence, Homeless, Income Shock
- Resilience: Lack Of Support Network, Low Emotional Resilience, Low Or Erratic Income, Over-indebtedness

TRANSCRIPT:
{transcript}

TASK:
Provide your assessment in the following JSON format:
{{
    "rating": "Medium/3",
    "vulnerability_types": ["Health: Mental Health Condition", "Resilience: Low Emotional Resilience"],
    "reasoning": "Detailed explanation of why you assigned this rating and identified these vulnerability types. Reference specific quotes from the transcript."
}}

Be specific and reference actual evidence from the transcript. Your reasoning should be clear and defensible.
"""


def call_claude_35_sonnet(transcript: str) -> Dict[str, Any]:
    """Call Claude 3.5 Sonnet via Bedrock for vulnerability assessment."""
    prompt = VULNERABILITY_ASSESSMENT_PROMPT.format(transcript=transcript)

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            body=json.dumps(request_body)
        )

        response_body = json.loads(response['body'].read())
        content = response_body['content'][0]['text']

        # Extract JSON from response
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = content[start_idx:end_idx]
            return json.loads(json_str)
        else:
            print(f"⚠ Claude response not in JSON format: {content[:200]}")
            return None

    except Exception as e:
        print(f"❌ Error calling Claude 3.5: {str(e)}")
        return None


def call_gpt4o(transcript: str, openai_api_key: str) -> Dict[str, Any]:
    """Call GPT-4o via OpenAI API for vulnerability assessment."""
    import openai

    openai.api_key = openai_api_key
    prompt = VULNERABILITY_ASSESSMENT_PROMPT.format(transcript=transcript)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert vulnerability assessor. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        print(f"❌ Error calling GPT-4o: {str(e)}")
        return None


def call_gemini_pro(transcript: str, google_api_key: str) -> Dict[str, Any]:
    """Call Gemini 2.5 Flash via Google AI API for vulnerability assessment."""
    import google.generativeai as genai

    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = VULNERABILITY_ASSESSMENT_PROMPT.format(transcript=transcript)

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
            )
        )

        content = response.text

        # Extract JSON from response
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = content[start_idx:end_idx]
            return json.loads(json_str)
        else:
            print(f"⚠ Gemini response not in JSON format: {content[:200]}")
            return None

    except Exception as e:
        print(f"❌ Error calling Gemini Pro: {str(e)}")
        return None


def get_transcript_from_s3(meeting_id: str, bucket_name: str) -> str:
    """Fetch redacted transcript from S3 using the partitioned structure."""
    s3 = boto3.client('s3', region_name=aws_region)

    # The transcripts are stored in partitioned structure
    # Pattern: summaries/supplementary/version=1.2/year=YYYY/month=MM/meeting_id=MEETING_ID/redacted_transcript.txt
    # We need to search for the meeting_id in the bucket

    prefix = f"summaries/supplementary/"

    try:
        # Use pagination to search for the meeting
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                # Check if this key contains our meeting_id and is a redacted transcript
                if f"meeting_id={meeting_id}" in key and key.endswith('redacted_transcript.txt'):
                    print(f"  ✓ Found redacted transcript at s3://{bucket_name}/{key}")
                    response = s3.get_object(Bucket=bucket_name, Key=key)
                    transcript = response['Body'].read().decode('utf-8')
                    return transcript

        print(f"  ⚠ Redacted transcript not found for meeting {meeting_id}")
        return None

    except Exception as e:
        print(f"  ❌ Error searching for transcript: {str(e)}")
        return None


def process_meeting(meeting_id: str, transcript: str, openai_api_key: str, google_api_key: str):
    """Process a single meeting through all AI models."""
    print(f"\n📊 Processing meeting {meeting_id}...")

    ai_responses = {}

    # Call Claude 3 Sonnet
    print("  🤖 Calling Claude 3 Sonnet...")
    claude_response = call_claude_35_sonnet(transcript)
    if claude_response:
        ai_responses['claude-3-sonnet'] = claude_response
        print(f"  ✓ Claude: {claude_response['rating']}")

    # Call GPT-4o
    print("  🤖 Calling GPT-4o...")
    gpt_response = call_gpt4o(transcript, openai_api_key)
    if gpt_response:
        ai_responses['gpt-4o'] = gpt_response
        print(f"  ✓ GPT-4o: {gpt_response['rating']}")

    # Call Gemini 2.5 Flash
    print("  🤖 Calling Gemini 2.5 Flash...")
    gemini_response = call_gemini_pro(transcript, google_api_key)
    if gemini_response:
        ai_responses['gemini-2.5-flash'] = gemini_response
        print(f"  ✓ Gemini: {gemini_response['rating']}")

    # Store in DynamoDB
    if ai_responses:
        try:
            assessments_table.update_item(
                Key={
                    'meeting_id': meeting_id,
                    'assessment_id': 'third-party'
                },
                UpdateExpression='SET ai_responses = :responses, ai_processed_at = :timestamp',
                ExpressionAttributeValues={
                    ':responses': ai_responses,
                    ':timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            print(f"  ✅ Stored {len(ai_responses)} AI responses in DynamoDB")
        except Exception as e:
            print(f"  ❌ Failed to store in DynamoDB: {str(e)}")
    else:
        print("  ⚠ No AI responses to store")


def main():
    parser = argparse.ArgumentParser(description='Generate AI vulnerability assessments')
    parser.add_argument('--meeting-id', help='Process a specific meeting ID')
    parser.add_argument('--limit', type=int, help='Limit number of meetings to process')
    parser.add_argument('--bucket', help='S3 bucket name for transcripts')
    args = parser.parse_args()

    # Get bucket from args or environment
    bucket = args.bucket or os.environ.get('S3_BUCKET', 'call-summariser-summarybucket-3wtnjhb9vvq0')

    # Get API keys from environment
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    google_api_key = os.environ.get('GOOGLE_API_KEY')

    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    if not google_api_key:
        print("❌ GOOGLE_API_KEY environment variable not set")
        sys.exit(1)

    print("🚀 Starting AI response generation...")
    print(f"   S3 Bucket: {bucket}")
    print(f"   AWS Region: {aws_region}")
    print(f"   Assessments Table: {assessments_table_name}")

    # Fetch meetings to process
    if args.meeting_id:
        meeting_ids = [args.meeting_id]
    else:
        # Fetch pending assessments from DynamoDB
        response = assessments_table.query(
            IndexName='ReviewStatusIndex',
            KeyConditionExpression=Key('review_status').eq('pending')
        )
        meeting_ids = [item['meeting_id'] for item in response['Items']]

        if args.limit:
            meeting_ids = meeting_ids[:args.limit]

    print(f"📋 Found {len(meeting_ids)} meetings to process")

    # Process each meeting
    for i, meeting_id in enumerate(meeting_ids, 1):
        print(f"\n[{i}/{len(meeting_ids)}] Processing {meeting_id}")

        # Get transcript
        transcript = get_transcript_from_s3(meeting_id, bucket)
        if not transcript:
            print(f"  ⏭ Skipping - no transcript available")
            continue

        # Process through AI models
        process_meeting(meeting_id, transcript, openai_api_key, google_api_key)

    print("\n✅ All meetings processed!")


if __name__ == '__main__':
    main()
