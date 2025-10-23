"""
Knowledge Base Retrieval Module for Case Check Examples
Uses AWS Bedrock Knowledge Base to retrieve relevant assessment examples
"""
import boto3
import json
from typing import List, Dict, Optional
from utils import helper
from constants import *

# Initialize Bedrock clients
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

# Knowledge Base Configuration
# KB_ID should be passed as parameter to functions (loaded from Parameter Store in app.py)
# Note: The Knowledge Base uses amazon.titan-embed-text-v2:0 for embeddings (configured in KB settings)


def retrieve_examples_for_check(
    check_id: str,
    check_description: str,
    max_results: int = 2,
    kb_id: Optional[str] = None
) -> List[Dict]:
    """
    Retrieve relevant assessment examples from Knowledge Base for a specific check.

    Args:
        check_id: The check identifier (e.g., "call_recording_confirmed")
        check_description: Human-readable description of what to check
        max_results: Maximum number of examples to retrieve (default: 2)
        kb_id: Knowledge Base ID (required - must be passed from caller)

    Returns:
        List of retrieved examples with content and metadata
    """
    if not kb_id:
        helper.log_json("WARNING", "KB_NOT_CONFIGURED",
                       message="Knowledge Base ID not configured, skipping retrieval")
        return []

    # Build search query optimized for retrieving relevant examples
    query = f"""
Find assessment examples for: {check_description}

Looking for examples that show:
- How to identify evidence for this check
- What constitutes Pass/Competent vs Fail
- Quality evidence quotes
- Appropriate comments and feedback

Check ID: {check_id}
""".strip()

    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={
                'text': query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results,
                    'overrideSearchType': 'HYBRID'  # Combines semantic + keyword search
                }
            }
        )

        results = response.get('retrievalResults', [])

        helper.log_json("INFO", "KB_RETRIEVAL_SUCCESS",
                       checkId=check_id,
                       numResults=len(results))

        # Extract and format results
        examples = []
        for result in results:
            content = result.get('content', {}).get('text', '')
            score = result.get('score', 0.0)
            metadata = result.get('metadata', {})

            examples.append({
                'content': content,
                'score': score,
                'metadata': metadata,
                'check_id': check_id
            })

        return examples

    except Exception as e:
        helper.log_json("ERROR", "KB_RETRIEVAL_FAILED",
                       checkId=check_id,
                       error=str(e))
        return []


def retrieve_examples_by_category(
    category: str,
    max_results: int = 3,
    kb_id: Optional[str] = None
) -> List[Dict]:
    """
    Retrieve examples for an entire category of checks (e.g., all compliance checks).

    Args:
        category: Category name ("compliance" or "macro")
        max_results: Maximum number of examples to retrieve
        kb_id: Knowledge Base ID (required - must be passed from caller)

    Returns:
        List of retrieved examples
    """
    if not kb_id:
        return []

    category_queries = {
        "compliance": "Show assessment examples for compliance criteria including call recording, regulated advice, personal details confirmation, and fees explanation",
        "macro": "Show assessment examples for coaching quality criteria including goal establishment, client engagement, and service explanation"
    }

    query = category_queries.get(category.lower(), "")
    if not query:
        helper.log_json("WARNING", "UNKNOWN_CATEGORY", category=category)
        return []

    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results,
                    'overrideSearchType': 'HYBRID'
                }
            }
        )

        results = response.get('retrievalResults', [])

        helper.log_json("INFO", "KB_CATEGORY_RETRIEVAL_SUCCESS",
                       category=category,
                       numResults=len(results))

        examples = []
        for result in results:
            examples.append({
                'content': result.get('content', {}).get('text', ''),
                'score': result.get('score', 0.0),
                'metadata': result.get('metadata', {}),
                'category': category
            })

        return examples

    except Exception as e:
        helper.log_json("ERROR", "KB_CATEGORY_RETRIEVAL_FAILED",
                       category=category,
                       error=str(e))
        return []


def format_examples_for_prompt(examples: List[Dict]) -> str:
    """
    Format retrieved examples into a string suitable for prompt injection.

    Args:
        examples: List of example dictionaries from KB retrieval

    Returns:
        Formatted string with examples
    """
    if not examples:
        return ""

    formatted = "\n\n--- REFERENCE EXAMPLES ---\n"
    formatted += "Use these examples to guide your assessment quality and format:\n\n"

    for idx, example in enumerate(examples, 1):
        content = example.get('content', '')
        score = example.get('score', 0.0)

        # Truncate very long examples to save tokens
        if len(content) > 1500:
            content = content[:1500] + "...[truncated]"

        formatted += f"EXAMPLE {idx} (relevance: {score:.2f}):\n"
        formatted += f"{content}\n"
        formatted += "-" * 60 + "\n\n"

    return formatted


def retrieve_and_format_examples(
    check_ids: List[str],
    check_descriptions: Dict[str, str],
    max_per_check: int = 1,
    kb_id: Optional[str] = None
) -> str:
    """
    Retrieve and format examples for multiple checks efficiently.

    Args:
        check_ids: List of check IDs to retrieve examples for
        check_descriptions: Dict mapping check_id to description
        max_per_check: Maximum examples per check
        kb_id: Knowledge Base ID

    Returns:
        Formatted examples string for prompt
    """
    # Strategy: Instead of querying for each check individually (expensive),
    # retrieve category-level examples that cover multiple checks

    compliance_checks = [
        "call_recording_confirmed", "regulated_advice_given", "vulnerability_identified",
        "dob_confirmed", "client_name_confirmed", "marital_status_confirmed",
        "citizenship_confirmed", "dependents_confirmed", "pension_details_confirmed",
        "income_expenditure_confirmed", "assets_liabilities_confirmed",
        "emergency_fund_confirmed", "will_confirmed", "fees_charges_explained",
        "way_forward_agreed"
    ]

    macro_checks = [
        "coach_introduction_signposting", "client_goals_established",
        "current_actions_established", "client_motivations_established",
        "relevant_suggestions_provided", "money_calculators_introduced",
        "asked_client_move_forward", "client_questions_opportunity"
    ]

    all_examples = []

    # Retrieve compliance examples (reduced to 1 for performance)
    has_compliance = any(c in compliance_checks for c in check_ids)
    if has_compliance:
        compliance_examples = retrieve_examples_by_category("compliance", max_results=1, kb_id=kb_id)
        all_examples.extend(compliance_examples)

    # Retrieve macro/quality examples (reduced to 1 for performance)
    has_macro = any(c in macro_checks for c in check_ids)
    if has_macro:
        macro_examples = retrieve_examples_by_category("macro", max_results=1, kb_id=kb_id)
        all_examples.extend(macro_examples)

    return format_examples_for_prompt(all_examples)
