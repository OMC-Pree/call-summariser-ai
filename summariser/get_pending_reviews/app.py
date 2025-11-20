"""
Lambda to fetch pending vulnerability assessments for coach review.
Returns all assessments with review_status='pending' AND assessment_type='vulnerability' from DynamoDB.
"""
import json
import os
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal

from utils import helper

dynamodb = boto3.resource('dynamodb')
table_name = os.environ.get('ASSESSMENTS_TABLE', 'vulnerability-assessments')
table = dynamodb.Table(table_name)


def decimal_default(obj):
    """JSON serializer for Decimal objects"""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def lambda_handler(event, context):
    """
    GET /reviews/pending

    Returns all pending vulnerability assessments that need coach review.
    Uses the ReviewStatusIndex GSI for efficient querying.

    Response: {
        count: int,
        assessments: [
            {
                meeting_id: str,
                assessment_id: str,      // e.g., "third-party#aveni", "vulnerability#123"
                overall_rating: str,
                vulnerability_types: [str],
                evidence_quotes: [str],
                assessed_at: str,
                model_consensus: {...},  // Optional
                ai_responses: {          // Optional
                    "gpt-4o": {...},
                    "gemini-pro": {...},
                    "claude-3.5-sonnet": {...}
                }
            }
        ]
    }
    """
    # Handle CORS preflight
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            },
            'body': ''
        }

    try:
        # Query pending vulnerability assessments
        # Try new optimized composite GSI first, fall back to old index if not ready
        try:
            # Optimized: Uses KeyConditionExpression on both review_status AND assessment_type
            response = table.query(
                IndexName='ReviewStatusTypeIndex',
                KeyConditionExpression=Key('review_status').eq('pending') & Key('assessment_type').eq('vulnerability')
            )
            helper.log_json("INFO", "USING_OPTIMIZED_GSI", index="ReviewStatusTypeIndex")
        except Exception as gsi_error:
            # Fallback: Use old index with FilterExpression (GSI may still be creating)
            helper.log_json("WARNING", "GSI_NOT_READY_FALLBACK",
                           index="ReviewStatusTypeIndex",
                           error=str(gsi_error),
                           message="Falling back to ReviewStatusIndex with FilterExpression")
            from boto3.dynamodb.conditions import Attr
            response = table.query(
                IndexName='ReviewStatusIndex',
                KeyConditionExpression=Key('review_status').eq('pending'),
                FilterExpression=Attr('assessment_type').eq('vulnerability')
            )

        assessments = response['Items']

        # Handle pagination if needed (DynamoDB returns max 1MB per query)
        while 'LastEvaluatedKey' in response:
            try:
                response = table.query(
                    IndexName='ReviewStatusTypeIndex',
                    KeyConditionExpression=Key('review_status').eq('pending') & Key('assessment_type').eq('vulnerability'),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
            except:
                # Fallback for pagination too
                from boto3.dynamodb.conditions import Attr
                response = table.query(
                    IndexName='ReviewStatusIndex',
                    KeyConditionExpression=Key('review_status').eq('pending'),
                    FilterExpression=Attr('assessment_type').eq('vulnerability'),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
            assessments.extend(response['Items'])

        # Transform to frontend-friendly format
        frontend_assessments = []
        for item in assessments:
            # Validate assessment_id exists (required for review submission)
            if 'assessment_id' not in item:
                helper.log_json("WARNING", "MISSING_ASSESSMENT_ID",
                               meeting_id=item.get('meeting_id'),
                               message="Assessment record missing assessment_id - data integrity issue")
                continue  # Skip this record

            # Only show third-party assessments that have AI comparison data
            # The review interface is for comparing third-party vs AI assessments
            assessment_id = item['assessment_id']
            if assessment_id.startswith('third-party#'):
                # Third-party assessment must have ai_responses for comparison
                if 'ai_responses' not in item or not item['ai_responses']:
                    helper.log_json("INFO", "SKIPPING_THIRD_PARTY_WITHOUT_AI",
                                   meeting_id=item.get('meeting_id'),
                                   assessment_id=assessment_id,
                                   message="Third-party assessment has no AI comparison data yet")
                    continue  # Skip - nothing to compare
            elif assessment_id.startswith('vulnerability#'):
                # AI-generated assessments are not shown in review interface
                # These are used for comparison, not for review
                continue  # Skip AI-only assessments

            assessment = {
                'meeting_id': item['meeting_id'],
                'assessment_id': item['assessment_id'],  # Required field
                # For vulnerability this will usually be something like "Critical"/"Medium"/etc.
                'overall_rating': item.get('result', item.get('vulnerability_rating', 'unknown')),
                'vulnerability_types': item.get('vulnerability_types', []),
                'evidence_quotes': item.get('evidence_quotes', []),
                'assessed_at': item.get('created_at', item.get('assessed_at', ''))
            }

            # Include multi-model consensus if available
            if 'model_consensus' in item:
                assessment['model_consensus'] = item['model_consensus']

            # Include AI model responses if available
            if 'ai_responses' in item:
                assessment['ai_responses'] = item['ai_responses']

            frontend_assessments.append(assessment)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({
                'count': len(frontend_assessments),
                'assessments': frontend_assessments
            }, default=decimal_default)
        }

    except Exception as e:
        print(f"Error fetching pending reviews: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',  # keep consistent with success path
                'Access-Control-Allow-Methods': 'GET,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({
                'error': 'Failed to fetch pending reviews',
                'details': str(e)
            })
        }
