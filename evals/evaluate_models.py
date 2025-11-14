"""
Script to evaluate AI model performance against coach ground truth reviews.

This script:
1. Fetches coach reviews (ground truth) from DynamoDB
2. Fetches AI model assessments from DynamoDB
3. Compares AI predictions vs ground truth
4. Calculates performance metrics for each model
5. Generates analysis and comparison report

Usage:
    python3 evaluate_models.py [--output results.json] [--html]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import boto3
from boto3.dynamodb.conditions import Key

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")

# Initialize AWS clients
aws_region = os.environ.get('AWS_REGION', 'eu-west-2')
dynamodb = boto3.resource('dynamodb', region_name=aws_region)

# Get table names from environment
assessments_table_name = os.environ.get('ASSESSMENTS_TABLE', 'vulnerability-assessments')
reviews_table_name = os.environ.get('REVIEWS_TABLE', 'vulnerability-ground-truth-reviews')

assessments_table = dynamodb.Table(assessments_table_name)
reviews_table = dynamodb.Table(reviews_table_name)


def extract_rating_number(rating: str) -> int:
    """Extract numeric rating from strings like 'Medium/3', 'High/4', etc."""
    if not rating:
        return 0
    parts = rating.split('/')
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def fetch_ground_truth_reviews() -> Dict[str, Dict]:
    """Fetch all coach reviews from DynamoDB."""
    print("📊 Fetching ground truth reviews...")

    reviews = {}

    try:
        response = reviews_table.scan()
    except Exception as e:
        if 'ResourceNotFoundException' in str(e):
            print("  ⚠ Reviews table not found. Deploy infrastructure with 'sam deploy' first.")
            return reviews
        raise

    for item in response['Items']:
        meeting_id = item['meeting_id']

        # Determine ground truth rating
        if item['action'] == 'agree':
            # Coach agreed with third-party assessment
            ground_truth_rating = item['third_party_rating']
            ground_truth_types = item.get('third_party_types', '')
        else:
            # Coach corrected the assessment
            ground_truth_rating = item.get('corrected_rating', item['third_party_rating'])
            ground_truth_types = item.get('corrected_types', item.get('third_party_types', ''))

        reviews[meeting_id] = {
            'rating': ground_truth_rating,
            'types': ground_truth_types,
            'action': item['action'],
            'coach_email': item['coach_email'],
            'timestamp': item.get('timestamp', '')
        }

    print(f"  ✓ Found {len(reviews)} ground truth reviews")
    return reviews


def fetch_ai_assessments() -> Dict[str, Dict]:
    """Fetch all AI assessments from DynamoDB."""
    print("🤖 Fetching AI assessments...")

    assessments = {}
    response = assessments_table.scan(
        FilterExpression='attribute_exists(ai_responses)'
    )

    for item in response['Items']:
        meeting_id = item['meeting_id']
        assessments[meeting_id] = {
            'ai_responses': item.get('ai_responses', {}),
            'third_party_rating': item.get('vulnerability_rating', ''),
            'third_party_types': item.get('vulnerability_types', [])
        }

    print(f"  ✓ Found {len(assessments)} meetings with AI assessments")
    return assessments


def calculate_metrics(predictions: List[int], ground_truth: List[int]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    if not predictions or not ground_truth:
        return {}

    # Exact accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions) if predictions else 0

    # Within-1 accuracy (off by at most 1 level)
    within_1 = sum(1 for p, g in zip(predictions, ground_truth) if abs(p - g) <= 1)
    within_1_accuracy = within_1 / len(predictions) if predictions else 0

    # Mean Absolute Error
    mae = sum(abs(p - g) for p, g in zip(predictions, ground_truth)) / len(predictions) if predictions else 0

    # Over/under prediction analysis
    over_predictions = sum(1 for p, g in zip(predictions, ground_truth) if p > g)
    under_predictions = sum(1 for p, g in zip(predictions, ground_truth) if p < g)

    return {
        'accuracy': accuracy,
        'within_1_accuracy': within_1_accuracy,
        'mae': mae,
        'over_predictions': over_predictions,
        'under_predictions': under_predictions,
        'exact_matches': correct,
        'total_predictions': len(predictions)
    }


def evaluate_models(ground_truth: Dict[str, Dict], assessments: Dict[str, Dict]) -> Dict[str, Any]:
    """Evaluate each AI model against ground truth."""
    print("\n📈 Evaluating model performance...")

    model_results = {
        'claude-3-sonnet': {'predictions': [], 'ground_truth': [], 'agreements': []},
        'gpt-4o': {'predictions': [], 'ground_truth': [], 'agreements': []},
        'gemini-2.5-flash': {'predictions': [], 'ground_truth': [], 'agreements': []}
    }

    # Also track third-party assessment performance
    third_party_results = {'predictions': [], 'ground_truth': [], 'agreements': []}

    # Collect predictions and ground truth for each model
    for meeting_id, gt in ground_truth.items():
        if meeting_id not in assessments:
            continue

        gt_rating = extract_rating_number(gt['rating'])
        assessment = assessments[meeting_id]

        # Third-party assessment
        tp_rating = extract_rating_number(assessment['third_party_rating'])
        third_party_results['predictions'].append(tp_rating)
        third_party_results['ground_truth'].append(gt_rating)
        third_party_results['agreements'].append(tp_rating == gt_rating)

        # AI model assessments
        ai_responses = assessment.get('ai_responses', {})
        for model_id, model_data in model_results.items():
            if model_id in ai_responses:
                ai_rating = extract_rating_number(ai_responses[model_id].get('rating', ''))
                model_data['predictions'].append(ai_rating)
                model_data['ground_truth'].append(gt_rating)
                model_data['agreements'].append(ai_rating == gt_rating)

    # Calculate metrics for each model
    results = {}

    # Third-party baseline
    results['third-party'] = {
        'metrics': calculate_metrics(
            third_party_results['predictions'],
            third_party_results['ground_truth']
        ),
        'name': 'Third-Party (Aveni)'
    }

    # AI models
    model_names = {
        'claude-3-sonnet': 'Claude 3 Sonnet',
        'gpt-4o': 'GPT-4o',
        'gemini-2.5-flash': 'Gemini 2.5 Flash'
    }

    for model_id, model_data in model_results.items():
        results[model_id] = {
            'metrics': calculate_metrics(
                model_data['predictions'],
                model_data['ground_truth']
            ),
            'name': model_names[model_id]
        }

    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results to console."""
    print("\n" + "=" * 80)
    print("📊 MODEL EVALUATION RESULTS")
    print("=" * 80)

    # Sort models by accuracy
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['metrics'].get('accuracy', 0),
        reverse=True
    )

    for rank, (model_id, data) in enumerate(sorted_models, 1):
        metrics = data['metrics']
        name = data['name']

        print(f"\n{rank}. {name} ({model_id})")
        print("-" * 80)

        if not metrics:
            print("  ⚠ No data available")
            continue

        print(f"  Exact Accuracy:      {metrics['accuracy']:.1%} ({metrics['exact_matches']}/{metrics['total_predictions']})")
        print(f"  Within-1 Accuracy:   {metrics['within_1_accuracy']:.1%}")
        print(f"  Mean Absolute Error: {metrics['mae']:.2f}")
        print(f"  Over-predictions:    {metrics['over_predictions']} ({metrics['over_predictions']/metrics['total_predictions']:.1%})")
        print(f"  Under-predictions:   {metrics['under_predictions']} ({metrics['under_predictions']/metrics['total_predictions']:.1%})")

    print("\n" + "=" * 80)


def generate_html_report(results: Dict[str, Any], output_path: str):
    """Generate HTML report with visualizations."""
    print(f"\n📄 Generating HTML report: {output_path}")

    # Sort models by accuracy
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['metrics'].get('accuracy', 0),
        reverse=True
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Evaluation Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .model-card {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .model-card.rank-1 {{ border-left-color: #27ae60; }}
        .model-card.rank-2 {{ border-left-color: #3498db; }}
        .model-card.rank-3 {{ border-left-color: #f39c12; }}
        .model-card.baseline {{ border-left-color: #95a5a6; }}

        .model-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .model-name {{
            font-size: 1.4em;
            font-weight: 700;
            color: #2c3e50;
        }}
        .rank-badge {{
            background: #27ae60;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metric {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: 700;
            color: #2c3e50;
        }}
        .metric-value.good {{ color: #27ae60; }}
        .metric-value.warning {{ color: #f39c12; }}
        .metric-value.bad {{ color: #e74c3c; }}

        .summary {{
            background: #e8f4f8;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            text-align: right;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Model Evaluation Results</h1>

        <div class="summary">
            <h3>Summary</h3>
            <p>This report compares the performance of different AI models against coach-validated ground truth labels for vulnerability assessment.</p>
            <ul>
                <li><strong>Total Evaluations:</strong> {sorted_models[0][1]['metrics'].get('total_predictions', 0)}</li>
                <li><strong>Models Evaluated:</strong> {len([m for m in sorted_models if m[0] != 'third-party'])}</li>
                <li><strong>Baseline:</strong> Third-Party (Aveni) Assessment</li>
            </ul>
        </div>
"""

    for rank, (model_id, data) in enumerate(sorted_models, 1):
        metrics = data['metrics']
        name = data['name']

        if not metrics:
            continue

        card_class = 'baseline' if model_id == 'third-party' else f'rank-{rank}'

        accuracy_class = 'good' if metrics['accuracy'] >= 0.7 else 'warning' if metrics['accuracy'] >= 0.5 else 'bad'

        html += f"""
        <div class="model-card {card_class}">
            <div class="model-header">
                <span class="model-name">{name}</span>
                {'<span class="rank-badge">🏆 Best Model</span>' if rank == 1 and model_id != 'third-party' else ''}
                {'<span class="rank-badge" style="background: #95a5a6;">Baseline</span>' if model_id == 'third-party' else ''}
            </div>

            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Exact Accuracy</div>
                    <div class="metric-value {accuracy_class}">{metrics['accuracy']:.1%}</div>
                    <div class="metric-label">{metrics['exact_matches']}/{metrics['total_predictions']} correct</div>
                </div>

                <div class="metric">
                    <div class="metric-label">Within-1 Accuracy</div>
                    <div class="metric-value">{metrics['within_1_accuracy']:.1%}</div>
                </div>

                <div class="metric">
                    <div class="metric-label">Mean Absolute Error</div>
                    <div class="metric-value">{metrics['mae']:.2f}</div>
                </div>

                <div class="metric">
                    <div class="metric-label">Over-predictions</div>
                    <div class="metric-value">{metrics['over_predictions']}</div>
                    <div class="metric-label">{metrics['over_predictions']/metrics['total_predictions']:.1%} of total</div>
                </div>

                <div class="metric">
                    <div class="metric-label">Under-predictions</div>
                    <div class="metric-value">{metrics['under_predictions']}</div>
                    <div class="metric-label">{metrics['under_predictions']/metrics['total_predictions']:.1%} of total</div>
                </div>
            </div>
        </div>
"""

    html += f"""
        <div class="timestamp">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"  ✓ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate AI model performance')
    parser.add_argument('--output', default='evaluation_results.json', help='Output JSON file path')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    args = parser.parse_args()

    print("🚀 Starting model evaluation...")
    print(f"   AWS Region: {aws_region}")
    print(f"   Assessments Table: {assessments_table_name}")
    print(f"   Reviews Table: {reviews_table_name}")

    # Fetch data
    ground_truth = fetch_ground_truth_reviews()
    assessments = fetch_ai_assessments()

    if not ground_truth:
        print("\n❌ No ground truth reviews found. Coaches need to review assessments first.")
        sys.exit(1)

    # Evaluate models
    results = evaluate_models(ground_truth, assessments)

    # Print results
    print_results(results)

    # Save JSON results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total_reviews': len(ground_truth),
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n💾 Results saved to {args.output}")

    # Generate HTML report if requested
    if args.html:
        html_path = args.output.replace('.json', '.html')
        generate_html_report(results, html_path)

    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
