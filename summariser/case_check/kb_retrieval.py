"""
Knowledge Base Retrieval Module for Case Check Examples
Uses AWS Bedrock Knowledge Base to retrieve relevant assessment examples
"""
import boto3
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from utils import helper
from constants import *

# Initialize Bedrock clients
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

# Knowledge Base Configuration
# KB_ID should be passed as parameter to functions (loaded from Parameter Store in app.py)
# Note: The Knowledge Base uses amazon.titan-embed-text-v2:0 for embeddings (configured in KB settings)

# Cache Configuration
# Cache category-level retrievals to avoid repeated API calls for the same content
# KB content is relatively static, so caching provides significant cost and latency savings
_CATEGORY_CACHE: Dict[str, tuple[datetime, List[Dict]]] = {}
_CACHE_DURATION = timedelta(hours=1)  # Adjust based on KB update frequency


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
    kb_id: Optional[str] = None,
    use_cache: bool = True
) -> List[Dict]:
    """
    Retrieve examples for an entire category of checks (e.g., all compliance checks).

    Uses in-memory caching to avoid repeated API calls for the same category.
    Cache expires after _CACHE_DURATION (default: 1 hour).

    Args:
        category: Category name ("compliance" or "macro")
        max_results: Maximum number of examples to retrieve
        kb_id: Knowledge Base ID (required - must be passed from caller)
        use_cache: Whether to use caching (default: True, set False for testing)

    Returns:
        List of retrieved examples
    """
    if not kb_id:
        return []

    # Check cache first
    if use_cache:
        cache_key = f"{category.lower()}_{max_results}_{kb_id}"
        cached = _CATEGORY_CACHE.get(cache_key)

        if cached:
            cache_time, cached_results = cached
            if datetime.now() - cache_time < _CACHE_DURATION:
                helper.log_json("INFO", "KB_CACHE_HIT",
                               category=category,
                               cacheAge=(datetime.now() - cache_time).seconds,
                               numResults=len(cached_results))
                return cached_results

    # Optimized queries for hybrid (semantic + keyword) search
    # - Frontload "Pass Fail" for critical context
    # - Include ALL check keywords for better matching
    # - Remove generic verbs ("show", "including") that dilute relevance
    # - Concise = better semantic density
    category_queries = {
        "compliance": (
            "Call assessment Pass Fail examples: "
            "call recording consent, regulated advice disclosure, vulnerability screening, "
            "client verification (DOB name marital status citizenship dependents), "
            "pension income expenditure assets liabilities emergency fund will, "
            "fees charges explanation, way forward agreement"
        ),
        "macro": (
            "Coaching assessment Pass Fail examples: "
            "coach introduction signposting, client goals motivations current actions, "
            "relevant suggestions money calculators, client questions move forward opportunity"
        )
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

        examples = []
        for result in results:
            examples.append({
                'content': result.get('content', {}).get('text', ''),
                'score': result.get('score', 0.0),
                'metadata': result.get('metadata', {}),
                'category': category
            })

        # Log retrieval quality metrics
        if examples:
            avg_score = sum(e['score'] for e in examples) / len(examples)
            total_chars = sum(len(e['content']) for e in examples)

            helper.log_json("INFO", "KB_CATEGORY_RETRIEVAL_SUCCESS",
                           category=category,
                           numResults=len(examples),
                           avgRelevanceScore=round(avg_score, 3),
                           totalChars=total_chars,
                           estimatedTokens=total_chars // 4)
        else:
            helper.log_json("INFO", "KB_CATEGORY_RETRIEVAL_SUCCESS",
                           category=category,
                           numResults=0)

        # Store in cache
        if use_cache and examples:
            cache_key = f"{category.lower()}_{max_results}_{kb_id}"
            _CATEGORY_CACHE[cache_key] = (datetime.now(), examples)

        return examples

    except Exception as e:
        helper.log_json("ERROR", "KB_CATEGORY_RETRIEVAL_FAILED",
                       category=category,
                       error=str(e),
                       errorType=type(e).__name__)
        return []


def format_examples_for_prompt(examples: List[Dict], max_tokens_per_example: int = 400) -> str:
    """
    Format retrieved examples with minimal token overhead.

    Uses token-aware truncation and removes verbose formatting to maximize
    the ratio of useful content to formatting overhead.

    Note: The instruction "Use these examples to guide assessment quality"
    should be in your main system prompt, not repeated here per call.

    Args:
        examples: List of example dictionaries from KB retrieval
        max_tokens_per_example: Rough token budget per example (default: 400)
                                Uses approximation of 1 token ≈ 4 characters

    Returns:
        Formatted string with examples (minimal overhead)
    """
    if not examples:
        return ""

    # Minimal header - use XML-style tags for clear boundaries
    formatted = "\n<examples>\n"

    for idx, example in enumerate(examples, 1):
        content = example.get('content', '')

        # Token-aware truncation (rough: 1 token ≈ 4 characters)
        max_chars = max_tokens_per_example * 4
        if len(content) > max_chars:
            # Truncate at sentence boundary for coherence
            truncated = content[:max_chars]
            last_period = truncated.rfind('.')
            # If we can keep >70% of content with sentence boundary, use it
            if last_period > max_chars * 0.7:
                content = truncated[:last_period + 1]
            else:
                content = truncated + "..."

        # Minimal formatting - no decorative separators or relevance scores
        # (LLMs don't effectively use relevance scores in practice)
        formatted += f"\n[Example {idx}]\n{content}\n"

    formatted += "</examples>\n"
    return formatted


def clear_cache():
    """
    Clear the KB retrieval cache.

    Call this function when:
    - Knowledge Base content has been updated
    - You want to force fresh retrievals for testing
    - Memory cleanup is needed
    """
    global _CATEGORY_CACHE
    _CATEGORY_CACHE.clear()
    helper.log_json("INFO", "KB_CACHE_CLEARED")


def get_cache_stats() -> Dict:
    """
    Get statistics about the current cache state.

    Returns:
        Dict with cache statistics including size, oldest/newest entries
    """
    if not _CATEGORY_CACHE:
        return {
            'size': 0,
            'entries': []
        }

    entries = []
    for key, (cache_time, results) in _CATEGORY_CACHE.items():
        age_seconds = (datetime.now() - cache_time).seconds
        entries.append({
            'key': key,
            'age_seconds': age_seconds,
            'num_results': len(results)
        })

    return {
        'size': len(_CATEGORY_CACHE),
        'entries': entries
    }


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
