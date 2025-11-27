#!/usr/bin/env python3
"""
Transform raw Aveni API data into unified DynamoDB assessment schema.

This script:
1. Maps Aveni classifier IDs to check_ids
2. Infers pass/fail from Aveni aiSummary text
3. Creates topic-level items for Aveni (12 checks per meeting)
4. Aggregates Aveni topics to overall assessment
5. Creates overall items for human assessments (reviewer only)
6. Outputs in unified DynamoDB schema format

Usage:
    # Dry run (preview transformation)
    python transform_golden_dataset.py --dry-run --limit 2

    # Transform all and output to file
    python transform_golden_dataset.py --output transformed_assessments.json

    # Transform and upload to DynamoDB
    python transform_golden_dataset.py --upload
"""

import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import argparse
from pathlib import Path


# Aveni Classifier ID to Check ID Mapping
# Based on actual classifier IDs from api_all_results.json
# Mapped to the 12 Phase 1 checks
CLASSIFIER_TO_CHECK_ID = {
    # Compliance checks (based on AI summary analysis)
    "6a77299e-d7ec-41f1-aaac-ebc3093ebdc5": "call_recording_confirmed",  # "call is being recorded for training and compliance"
    "f9e9a4cf-b745-41ea-9d4a-5dcbc8f13049": "date_of_birth_confirmed",  # "asks for the customer's date of birth"
    "bae55373-a9fa-452c-84a3-bc5ad309f382": "date_of_birth_confirmed",  # Alternate DOB check (references client's DOB)
    "10b645d0-c9f4-4790-99d7-3a479b5a736a": "client_name_confirmed",  # "refers to the client's name when reviewing details"
    "50b88552-53d5-4ce4-bd37-bc13d617870e": "client_name_confirmed",  # "asked for their full legal name"
    "5b24f4f7-45ba-4ced-abbb-5b944d94c302": "marital_status_confirmed",  # "discuss the client's partner... does not directly confirm marital status"
    "bf42b094-ce53-4f29-a1b2-bc26f2a5f518": "assets_liabilities_confirmed",  # "covers both assets and liabilities"
    "2d0a02fd-b585-4fbd-ab85-e244f3f49400": "emergency_fund_confirmed",  # "detailed discussion about client's emergency fund"
    "f8933a6b-3ad3-4be4-a71c-e9b8301fc857": "fees_explained",  # "clear explanation of fees and charges"
    "3cbbffd7-82de-48a3-b43f-90dddb94eeed": "way_forward_agreed",  # "scheduling a follow-up meeting"

    # Macro checks
    "713cae3c-5124-4188-9294-72abb3f6bab5": "coach_introduction",  # "coach introduced Octopus Money... signposted structure"
    "7e0a080c-f2a2-44a9-ab1d-1f244a8b0cb2": "asked_to_move_forward",  # "asked client for thoughts on moving forward"
    "f088087e-e55a-4a40-8ad3-2f1096ca56ef": "opportunity_to_ask_questions",  # "provided client opportunity to ask questions"
    "f27c5a2d-040b-4d5c-bd48-664014414a29": "vulnerability_identified",  # "customer mentioned feeling 'slightly anxious' about money"
}

# Reverse mapping for validation
CHECK_ID_TO_CLASSIFIER = {v: k for k, v in CLASSIFIER_TO_CHECK_ID.items()}


def infer_pass_fail(ai_summary: str) -> Tuple[str, str]:
    """
    Infer pass/fail result from Aveni AI summary text.

    Returns:
        Tuple of (result: "pass"/"fail", confidence: "high"/"medium"/"low")
    """
    if not ai_summary or not isinstance(ai_summary, str):
        return "fail", "low"

    summary_lower = ai_summary.lower()

    # Strong pass indicators
    strong_pass = [
        "confirmed",
        "verified",
        "established",
        "discussed thoroughly",
        "clearly explained",
        "adequately addressed",
    ]

    # Strong fail indicators
    strong_fail = [
        "not confirmed",
        "not verified",
        "not established",
        "not discussed",
        "not explained",
        "missing",
        "unclear",
        "did not",
        "failed to",
        "no evidence",
    ]

    # Weak/development indicators (count as fail for Phase 1)
    weak_indicators = [
        "could be improved",
        "should have",
        "would benefit",
        "development needed",
        "assumed",
    ]

    # Check for strong fail first
    for indicator in strong_fail:
        if indicator in summary_lower:
            return "fail", "high"

    # Check for weak indicators
    for indicator in weak_indicators:
        if indicator in summary_lower:
            return "fail", "medium"

    # Check for strong pass
    for indicator in strong_pass:
        if indicator in summary_lower:
            return "pass", "high"

    # Default: if no clear indicators, assume pass but low confidence
    # (Aveni tends to highlight problems, so silence = likely ok)
    return "pass", "low"


def extract_check_evidence(ai_summary: str, max_length: int = 500) -> str:
    """
    Extract relevant evidence snippet from AI summary.
    Truncate if too long to fit DynamoDB item limits.
    """
    if not ai_summary:
        return ""

    # Take first sentence or max_length chars, whichever is shorter
    summary = ai_summary.strip()
    if len(summary) <= max_length:
        return summary

    # Try to find a sentence boundary
    truncated = summary[:max_length]
    last_period = truncated.rfind('. ')
    if last_period > max_length * 0.5:  # If we can get at least half
        return truncated[:last_period + 1]

    return truncated + "..."


def create_topic_item_aveni(
    meeting_id: str,
    case_id: str,
    check_id: str,
    classifier_id: str,
    ai_summary: str,
    created_at: str,
) -> Dict:
    """
    Create a topic-level DynamoDB item for Aveni AI assessment.

    Pattern: case-check#topic#{check_id}#thirdparty-ai
    """
    result, confidence = infer_pass_fail(ai_summary)
    evidence = extract_check_evidence(ai_summary)

    return {
        "meeting_id": meeting_id,
        "assessment_id": f"case-check#topic#{check_id}#thirdparty-ai",

        # Classification fields
        "assessment_type": "case-check",
        "aggregation_level": "topic",
        "check_id": check_id,
        "source": "thirdparty-ai",
        "source_type": "third-party",

        # Assessment result
        "check_result": result,
        "result_confidence": confidence,
        "check_evidence": evidence,

        # Metadata
        "case_id": case_id,
        "classifier_id": classifier_id,  # For debugging/validation
        "created_at": created_at,
        "review_status": "completed",
    }


def create_overall_item_aveni(
    meeting_id: str,
    case_id: str,
    topic_items: List[Dict],
    created_at: str,
) -> Dict:
    """
    Create an overall-level DynamoDB item for Aveni AI assessment.
    Aggregates from topic-level results.

    Pattern: case-check#overall#thirdparty-ai
    """
    total_checks = len(topic_items)
    passed_checks = sum(1 for item in topic_items if item["check_result"] == "pass")
    failed_checks = total_checks - passed_checks

    overall_score = Decimal(str(round((passed_checks / total_checks * 100), 2))) if total_checks > 0 else Decimal("0")
    overall_result = "pass" if passed_checks == total_checks else "fail"

    return {
        "meeting_id": meeting_id,
        "assessment_id": "case-check#overall#thirdparty-ai",

        # Classification fields
        "assessment_type": "case-check",
        "aggregation_level": "overall",
        "source": "thirdparty-ai",
        "source_type": "third-party",

        # Overall assessment result
        "overall_score": overall_score,
        "overall_result": overall_result,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,

        # Metadata
        "case_id": case_id,
        "created_at": created_at,
        "review_status": "completed",
    }


def create_overall_item_human(
    meeting_id: str,
    case_id: str,
    reviewer_evaluation: Dict,
    created_at: str,
) -> Dict:
    """
    Create an overall-level DynamoDB item for human assessment.
    Uses only the reviewer evaluation (Eval #2) as ground truth.

    Pattern: case-check#overall#thirdparty-human
    """
    score_data = reviewer_evaluation.get("score", {})
    outcome_data = reviewer_evaluation.get("outcome", {})

    points = Decimal(str(score_data.get("points", 0)))
    maximum = Decimal(str(score_data.get("maximum", 26)))

    overall_score = Decimal(str(round((points / maximum * 100), 2))) if maximum > 0 else Decimal("0")
    overall_result = outcome_data.get("grade", "unknown").lower()

    # Extract feedback
    comment = reviewer_evaluation.get("comment", "")

    # Try to parse "Things done well:" and "Things to work on:" from comment
    feedback = {
        "things_done_well": [],
        "things_to_work_on": [],
        "raw_comment": comment,
    }

    if comment:
        # Simple parsing - can be enhanced later
        if "Things done well:" in comment:
            well_section = comment.split("Things done well:")[1]
            if "Things to work on:" in well_section:
                well_section = well_section.split("Things to work on:")[0]
            feedback["things_done_well"] = [
                line.strip("- ").strip()
                for line in well_section.split("\n")
                if line.strip() and line.strip() != "Things done well:"
            ]

        if "Things to work on:" in comment:
            work_section = comment.split("Things to work on:")[1]
            feedback["things_to_work_on"] = [
                line.strip("- ").strip()
                for line in work_section.split("\n")
                if line.strip()
            ]

    return {
        "meeting_id": meeting_id,
        "assessment_id": "case-check#overall#thirdparty-human",

        # Classification fields
        "assessment_type": "case-check",
        "aggregation_level": "overall",
        "source": "thirdparty-human",
        "source_type": "third-party",

        # Overall assessment result
        "overall_score": overall_score,
        "overall_result": overall_result,
        "score_points": points,
        "score_maximum": maximum,

        # Human feedback
        "feedback": feedback,

        # Metadata
        "case_id": case_id,
        "reviewer_role": reviewer_evaluation.get("role", "reviewer"),
        "created_at": created_at,
        "review_status": "completed",
    }


def transform_case(case_data: Dict, verbose: bool = False) -> List[Dict]:
    """
    Transform a single case from api_all_results.json into DynamoDB items.

    Returns:
        List of DynamoDB items (topic + overall for each source)
    """
    items = []

    meeting_id = case_data.get("meeting_id")
    case_id = case_data.get("case_id")

    if not meeting_id or not case_id:
        if verbose:
            print(f"⚠️  Skipping case - missing meeting_id or case_id: {case_data}")
        return items

    created_at = datetime.now(timezone.utc).isoformat()

    # Process Aveni AI data (topic-level)
    aveni_data = case_data.get("aveni_data", {})
    if aveni_data and aveni_data.get("data"):
        ai_summary_list = aveni_data["data"].get("aiSummaryList", [])
        aveni_topic_items = []

        for summary_item in ai_summary_list:
            classifier_id = summary_item.get("classifierId")
            check_id = CLASSIFIER_TO_CHECK_ID.get(classifier_id)

            if not check_id:
                if verbose:
                    print(f"⚠️  Unknown classifier ID: {classifier_id} for meeting {meeting_id}")
                continue

            ai_summary = summary_item.get("aiSummary", "")

            topic_item = create_topic_item_aveni(
                meeting_id=meeting_id,
                case_id=case_id,
                check_id=check_id,
                classifier_id=classifier_id,
                ai_summary=ai_summary,
                created_at=created_at,
            )

            aveni_topic_items.append(topic_item)
            items.append(topic_item)

        # Create Aveni overall item (aggregated from topics)
        if aveni_topic_items:
            overall_item = create_overall_item_aveni(
                meeting_id=meeting_id,
                case_id=case_id,
                topic_items=aveni_topic_items,
                created_at=created_at,
            )
            items.append(overall_item)

    # Process Human assessment data (overall-level only)
    human_data = case_data.get("human_data", {})
    if human_data and human_data.get("data"):
        evaluations = human_data["data"].get("evaluations", [])

        # Find reviewer evaluation (Eval #2)
        reviewer_eval = None
        for evaluation in evaluations:
            if evaluation.get("role") == "reviewer":
                reviewer_eval = evaluation
                break

        if reviewer_eval:
            overall_item = create_overall_item_human(
                meeting_id=meeting_id,
                case_id=case_id,
                reviewer_evaluation=reviewer_eval,
                created_at=created_at,
            )
            items.append(overall_item)
        elif verbose:
            print(f"⚠️  No reviewer evaluation found for meeting {meeting_id}")

    return items


def transform_all_cases(
    input_file: Path,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict]:
    """
    Transform all cases from api_all_results.json.

    Returns:
        List of all DynamoDB items ready for upload
    """
    print(f"📖 Loading data from {input_file}...")

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Handle both formats: direct array or {"cases": [...]}
    if isinstance(data, list):
        cases = data
    else:
        cases = data.get("cases", [])

    total_cases = len(cases)

    if limit:
        cases = cases[:limit]
        print(f"🔢 Processing {len(cases)} out of {total_cases} cases (limit={limit})")
    else:
        print(f"🔢 Processing all {total_cases} cases")

    all_items = []
    stats = {
        "cases_processed": 0,
        "cases_with_aveni": 0,
        "cases_with_human": 0,
        "aveni_topic_items": 0,
        "aveni_overall_items": 0,
        "human_overall_items": 0,
    }

    for i, case_data in enumerate(cases, 1):
        if verbose:
            print(f"\n--- Processing case {i}/{len(cases)}: {case_data.get('meeting_id')} ---")

        items = transform_case(case_data, verbose=verbose)

        if items:
            stats["cases_processed"] += 1

            # Count item types
            for item in items:
                assessment_id = item["assessment_id"]
                if "thirdparty-ai" in assessment_id:
                    if "#topic#" in assessment_id:
                        stats["aveni_topic_items"] += 1
                        stats["cases_with_aveni"] = stats.get("cases_with_aveni", 0) + (1 if stats["aveni_topic_items"] == 1 else 0)
                    elif "#overall#" in assessment_id:
                        stats["aveni_overall_items"] += 1
                elif "thirdparty-human" in assessment_id:
                    stats["human_overall_items"] += 1
                    stats["cases_with_human"] += 1

            all_items.extend(items)

    print(f"\n✅ Transformation complete!")
    print(f"\n📊 Statistics:")
    print(f"   Cases processed: {stats['cases_processed']}")
    print(f"   Cases with Aveni data: {stats['cases_with_aveni']}")
    print(f"   Cases with Human data: {stats['cases_with_human']}")
    print(f"   Aveni topic items: {stats['aveni_topic_items']}")
    print(f"   Aveni overall items: {stats['aveni_overall_items']}")
    print(f"   Human overall items: {stats['human_overall_items']}")
    print(f"   Total items: {len(all_items)}")

    return all_items


def main():
    parser = argparse.ArgumentParser(
        description="Transform raw Aveni API data into unified DynamoDB assessment schema"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "golden_dataset_results" / "api_all_results.json",
        help="Input JSON file (default: golden_dataset_results/api_all_results.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for transformed items",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview transformation without writing output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of cases to process (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed transformation logs",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload transformed items to DynamoDB (NOT IMPLEMENTED YET)",
    )

    args = parser.parse_args()

    # Transform all cases
    items = transform_all_cases(
        input_file=args.input,
        limit=args.limit,
        verbose=args.verbose,
    )

    # Dry-run mode: show sample items
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - Sample items:")
        print(f"\n--- Sample Aveni Topic Item ---")
        sample_topic = next((item for item in items if "#topic#" in item["assessment_id"]), None)
        if sample_topic:
            print(json.dumps(sample_topic, indent=2, default=str))

        print(f"\n--- Sample Aveni Overall Item ---")
        sample_aveni_overall = next((item for item in items if "thirdparty-ai" in item["assessment_id"] and "#overall#" in item["assessment_id"]), None)
        if sample_aveni_overall:
            print(json.dumps(sample_aveni_overall, indent=2, default=str))

        print(f"\n--- Sample Human Overall Item ---")
        sample_human_overall = next((item for item in items if "thirdparty-human" in item["assessment_id"]), None)
        if sample_human_overall:
            print(json.dumps(sample_human_overall, indent=2, default=str))

        print(f"\n💡 Use --output to save transformed items to a file")
        return

    # Write output file
    if args.output:
        print(f"\n💾 Writing transformed items to {args.output}...")
        with open(args.output, 'w') as f:
            json.dump({
                "metadata": {
                    "transformed_at": datetime.now(timezone.utc).isoformat(),
                    "source_file": str(args.input),
                    "total_items": len(items),
                },
                "items": items,
            }, f, indent=2, default=str)
        print(f"✅ Saved {len(items)} items to {args.output}")

    # Upload to DynamoDB
    if args.upload:
        print(f"\n⚠️  DynamoDB upload not implemented yet")
        print(f"💡 Use --output to save to JSON first, then implement batch upload")


if __name__ == "__main__":
    main()
