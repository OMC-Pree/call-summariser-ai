"""
Lambda to save vulnerability ground truth validation reviews to DynamoDB.
Updates assessment record with coach review decision (single-table design).
"""
import json
import os
import boto3
from datetime import datetime

dynamodb = boto3.resource('dynamodb')
assessments_table_name = os.environ.get('ASSESSMENTS_TABLE', 'vulnerability-assessments')
assessments_table = dynamodb.Table(assessments_table_name)


def lambda_handler(event, context):
    """
    POST /review
    Body: {
        action: 'agree' | 'correct',
        meeting_id: str,
        third_party_rating: str,
        third_party_types: str,
        corrected_rating?: str,
        corrected_types?: str,
        reasoning?: str,
        coach_email: str,
        timestamp: str (ISO)
    }

    Updates the vulnerability-assessments record with:
    - review_status: "reviewed"
    - reviewed_by: coach_email
    - reviewed_at: timestamp
    - coach_action: "agree" | "correct"
    - coach_corrected_rating (if action=correct)
    - coach_corrected_types (if action=correct)
    - coach_reasoning (if action=correct)
    """
    # Handle CORS preflight
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            },
            'body': ''
        }

    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))

        # Validate required fields
        required_fields = ['action', 'meeting_id', 'third_party_rating',
                          'third_party_types', 'coach_email', 'timestamp']
        for field in required_fields:
            if field not in body:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'POST,OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type'
                    },
                    'body': json.dumps({
                        'error': f'Missing required field: {field}'
                    })
                }

        # Build update expression based on action
        update_expression = 'SET review_status = :status, reviewed_by = :coach, reviewed_at = :timestamp, coach_action = :action'
        expression_values = {
            ':status': 'reviewed',
            ':coach': body['coach_email'],
            ':timestamp': datetime.utcnow().isoformat(),
            ':action': body['action']
        }

        # Add correction fields if action is 'correct'
        if body['action'] == 'correct':
            update_expression += ', coach_corrected_rating = :corrected_rating, coach_corrected_types = :corrected_types, coach_reasoning = :reasoning'
            expression_values[':corrected_rating'] = body.get('corrected_rating', '')
            expression_values[':corrected_types'] = body.get('corrected_types', '')
            expression_values[':reasoning'] = body.get('reasoning', '')

        # Update assessment record
        assessments_table.update_item(
            Key={
                'meeting_id': body['meeting_id'],
                'assessment_id': 'third-party'
            },
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values
        )

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({
                'message': 'Review saved successfully',
                'meeting_id': body['meeting_id'],
                'assessment_status': 'reviewed'
            })
        }

    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': 'http://coach-review-interface-vulnerability.s3-website.eu-west-2.amazonaws.com'
            },
            'body': json.dumps({'error': 'Invalid JSON in request body'})
        }
    except Exception as e:
        print(f"Error saving review: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': 'http://coach-review-interface-vulnerability.s3-website.eu-west-2.amazonaws.com'
            },
            'body': json.dumps({
                'error': 'Failed to save review',
                'details': str(e)
            })
        }
