"""
Lambda to fetch pending vulnerability assessments for coach review.
Returns all assessments with review_status='pending' from DynamoDB.
"""
import json
import os
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table_name = os.environ.get('ASSESSMENTS_TABLE', 'vulnerability-assessments')
table = dynamodb.Table(table_name)


def decimal_default(obj):
    """JSON serializer for Decimal objects"""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    raise TypeError


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
                overall_rating: str,
                vulnerability_types: [str],
                evidence_quotes: [str],
                assessed_at: str,
                model_consensus: {...},  // Optional
                ai_responses: {          // Optional
                    "gpt-4o": {
                        rating: str,
                        vulnerability_types: [str],
                        reasoning: str
                    },
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
        # Query pending assessments using GSI
        response = table.query(
            IndexName='ReviewStatusIndex',
            KeyConditionExpression=Key('review_status').eq('pending'),
            ScanIndexForward=False  # Most recent first (by assessed_at)
        )

        assessments = response['Items']

        # Handle pagination if needed (DynamoDB returns max 1MB per query)
        while 'LastEvaluatedKey' in response:
            response = table.query(
                IndexName='ReviewStatusIndex',
                KeyConditionExpression=Key('review_status').eq('pending'),
                ScanIndexForward=False,
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            assessments.extend(response['Items'])

        # Transform to frontend-friendly format
        frontend_assessments = []
        for item in assessments:
            assessment = {
                'meeting_id': item['meeting_id'],
                'overall_rating': item['vulnerability_rating'],
                'vulnerability_types': item['vulnerability_types'],
                'evidence_quotes': item['evidence_quotes'],
                'assessed_at': item['assessed_at']
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
                'Access-Control-Allow-Origin': 'http://coach-review-interface-vulnerability.s3-website.eu-west-2.amazonaws.com'
            },
            'body': json.dumps({
                'error': 'Failed to fetch pending reviews',
                'details': str(e)
            })
        }
