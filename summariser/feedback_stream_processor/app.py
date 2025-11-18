"""
Feedback Stream Processor Lambda
Processes DynamoDB Streams from assessment-results table.
When review_status changes to 'reviewed', writes training data to S3 JSONL.
"""
import json
import boto3
import os
from datetime import datetime
from uuid import uuid4
from decimal import Decimal
from typing import Dict, Optional, Any
from boto3.dynamodb.types import TypeDeserializer

s3 = boto3.client('s3')
GOLDEN_DATA_BUCKET = os.environ['GOLDEN_DATA_BUCKET']
SUMMARY_BUCKET = os.environ['SUMMARY_BUCKET']

# DynamoDB type deserializer
deserializer = TypeDeserializer()


def deserialize_dynamodb_item(item: Dict) -> Dict:
    """Convert DynamoDB item format to Python dict"""
    return {k: deserializer.deserialize(v) for k, v in item.items()}


def should_process_record(old_image: Optional[Dict], new_image: Dict) -> bool:
    """Check if this record should be processed (review status changed to 'reviewed')"""
    new_status = new_image.get('review_status')

    if new_status != 'reviewed':
        return False

    # Don't process if already was reviewed
    if old_image:
        old_status = old_image.get('review_status')
        if old_status == 'reviewed':
            return False

    return True


def fetch_transcript(s3_key: str) -> str:
    """Fetch transcript from S3"""
    try:
        response = s3.get_object(Bucket=SUMMARY_BUCKET, Key=s3_key)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        print(f"ERROR fetching transcript from {s3_key}: {e}")
        return ""


def build_training_example(assessment: Dict, transcript: str) -> Dict:
    """
    Build ML training example from assessment and transcript.

    Format depends on assessment_type:
    - summary: {input: transcript, output: summary_text, metadata: {...}}
    - case-check: {input: transcript, check: {...}, metadata: {...}}
    - vulnerability: {input: transcript, vulnerability: {...}, metadata: {...}}
    """
    assessment_type = assessment['assessment_type']
    coach_action = assessment.get('coach_action', 'agree')

    # Parse AI output
    ai_output = json.loads(assessment['ai_output']) if isinstance(assessment['ai_output'], str) else assessment['ai_output']

    # Determine ground truth (coach correction OR original AI output)
    if coach_action == 'correct' and assessment.get('coach_corrected_output'):
        ground_truth = json.loads(assessment['coach_corrected_output']) if isinstance(assessment['coach_corrected_output'], str) else assessment['coach_corrected_output']
    else:
        ground_truth = ai_output

    # Base metadata
    metadata = {
        'meeting_id': assessment['meeting_id'],
        'assessment_id': assessment['assessment_id'],
        'ai_version': assessment.get('ai_version', 'unknown'),
        'model_name': assessment.get('model_name', 'unknown'),
        'coach_email': assessment.get('reviewed_by', 'unknown'),
        'reviewed_at': assessment.get('reviewed_at', ''),
        'feedback_type': coach_action,
        'coach_confidence': assessment.get('coach_confidence', 'medium'),
        'coach_reasoning': assessment.get('coach_reasoning', ''),
        'session_type': assessment.get('session_type', 'unknown'),
        'created_at': assessment.get('created_at', '')
    }

    # Build type-specific training example
    if assessment_type == 'summary':
        return {
            'input': transcript,
            'output': ground_truth,
            'metadata': {
                **metadata,
                'quality_score': float(assessment.get('quality_score', 0)),  # From top-level DynamoDB field
                'ai_quality_score': float(assessment.get('quality_score', 0))  # Same since we don't store corrections
            }
        }

    elif assessment_type == 'case-check':
        check_id = assessment.get('check_id')
        if not check_id:
            parts = assessment.get('assessment_id', '').split('#')
            check_id = parts[1] if len(parts) > 1 else 'unknown'

        return {
            'input': transcript,
            'check': {
                'id': check_id,
                'result': ground_truth.get('result', ground_truth.get('status', 'UNKNOWN')),
                'evidence': ground_truth.get('evidence', ground_truth.get('evidence_quote', '')),
                'confidence': float(ai_output.get('confidence', 0)),
                'human_verified': (coach_action == 'agree'),
                'ai_result': ai_output.get('result', ai_output.get('status', 'UNKNOWN')),
                'coach_correction': ground_truth if coach_action == 'correct' else None
            },
            'metadata': metadata
        }

    elif assessment_type == 'vulnerability':
        return {
            'input': transcript,
            'vulnerability': {
                'ai_rating': ai_output.get('rating', ai_output.get('vulnerability_rating', 'UNKNOWN')),
                'ai_types': ai_output.get('types', ai_output.get('vulnerability_types', [])),
                'human_rating': ground_truth.get('rating', ground_truth.get('vulnerability_rating', 'UNKNOWN')),
                'human_types': ground_truth.get('types', ground_truth.get('vulnerability_types', [])),
                'indicators': ground_truth.get('indicators', ground_truth.get('vulnerability_indicators', [])),
                'recommended_actions': ground_truth.get('recommended_actions', []),
                'human_verified': (coach_action == 'agree'),
                'severity_changed': (
                    ai_output.get('rating', ai_output.get('vulnerability_rating')) !=
                    ground_truth.get('rating', ground_truth.get('vulnerability_rating'))
                )
            },
            'metadata': metadata
        }

    else:
        # Unknown type - return basic format
        return {
            'input': transcript,
            'output': ground_truth,
            'metadata': metadata
        }


def convert_decimals(obj: Any) -> Any:
    """Convert Decimal objects to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    return obj


def append_to_jsonl(assessment_type: str, training_example: Dict):
    """
    Append training example to S3 JSONL file.
    File structure: training-data/{type}/year={YYYY}/month={MM}/labeled_{type}.jsonl
    """
    now = datetime.utcnow()
    s3_key = f"training-data/{assessment_type}/year={now.year}/month={now.month:02d}/labeled_{assessment_type}.jsonl"

    # Convert Decimals to float for JSON serialization
    training_example = convert_decimals(training_example)

    # Try to fetch existing file
    try:
        response = s3.get_object(Bucket=GOLDEN_DATA_BUCKET, Key=s3_key)
        existing_content = response['Body'].read().decode('utf-8')
    except s3.exceptions.NoSuchKey:
        existing_content = ''
    except Exception as e:
        print(f"ERROR fetching existing JSONL file {s3_key}: {e}")
        existing_content = ''

    # Append new example
    new_line = json.dumps(training_example) + '\n'
    updated_content = existing_content + new_line

    # Write back to S3
    try:
        s3.put_object(
            Bucket=GOLDEN_DATA_BUCKET,
            Key=s3_key,
            Body=updated_content.encode('utf-8'),
            ContentType='application/x-ndjson'
        )
        print(f"✅ Appended training example to {s3_key}")
        return s3_key
    except Exception as e:
        print(f"ERROR writing to S3 {s3_key}: {e}")
        raise


def lambda_handler(event, context):
    """
    Process DynamoDB Stream records.
    When review_status changes to 'reviewed', create training data.
    """
    print(f"Received {len(event['Records'])} stream records")

    processed_count = 0
    error_count = 0

    for record in event['Records']:
        try:
            # Only process MODIFY events (coach reviews are updates)
            if record['eventName'] != 'MODIFY':
                continue

            # Deserialize DynamoDB items
            old_image = deserialize_dynamodb_item(record['dynamodb'].get('OldImage', {})) if 'OldImage' in record['dynamodb'] else None
            new_image = deserialize_dynamodb_item(record['dynamodb']['NewImage'])

            # Check if we should process this record
            if not should_process_record(old_image, new_image):
                continue

            # Extract data
            meeting_id = new_image['meeting_id']
            assessment_id = new_image['assessment_id']
            assessment_type = new_image['assessment_type']
            transcript_key = new_image.get('transcript_s3_key')

            if not transcript_key:
                print(f"WARNING: No transcript_s3_key for {meeting_id}/{assessment_id}, skipping")
                continue

            print(f"Processing review for {meeting_id}/{assessment_id} (type: {assessment_type})")

            # Fetch transcript
            transcript = fetch_transcript(transcript_key)
            if not transcript:
                print(f"WARNING: Empty transcript for {transcript_key}, skipping")
                continue

            # Build training example
            training_example = build_training_example(new_image, transcript)

            # Append to S3 JSONL
            s3_key = append_to_jsonl(assessment_type, training_example)

            print(f"✅ Processed {assessment_type} training example for {meeting_id}")
            processed_count += 1

        except Exception as e:
            error_count += 1
            print(f"ERROR processing record: {e}")
            import traceback
            traceback.print_exc()
            # Continue processing other records

    print(f"Processed {processed_count} records successfully, {error_count} errors")

    return {
        'statusCode': 200,
        'processed': processed_count,
        'errors': error_count
    }
