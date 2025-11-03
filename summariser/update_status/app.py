"""
Update Status Lambda - Step Functions workflow step
Updates job status in DynamoDB with metadata
"""
import boto3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from utils.error_handler import lambda_error_handler, ValidationError
from constants import *

dynamodb = boto3.resource("dynamodb")
JOB_TABLE_REF = dynamodb.Table(SUMMARY_JOB_TABLE)


def _to_ddb_numbers(obj):
    """
    Recursively convert all Python floats in obj to Decimal for DynamoDB.
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_ddb_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_ddb_numbers(v) for v in obj]
    return obj


@lambda_error_handler()
def lambda_handler(event, context):
    """
    Update job status in DynamoDB.

    Input:
        - meetingId: str
        - status: str (QUEUED|PROCESSING|COMPLETED|FAILED|IN_REVIEW)
        - metadata: dict (optional additional fields)
        - force: bool (optional, default False)
        - workflowType: str (optional: "summary" or "case_check")

    Output:
        - success: bool
        - status: str
    """
    meeting_id = event.get("meetingId")
    status = event.get("status")
    metadata = event.get("metadata", {})
    force = event.get("force", False)
    workflow_type = event.get("workflowType", "summary")  # Default to summary for backward compatibility

    if not meeting_id:
        raise ValidationError("meetingId is required")

    if not status:
        raise ValidationError("status is required")

    status_up = status.upper()
    now_iso = datetime.now(timezone.utc).isoformat()
    exp_ts = int((datetime.now(timezone.utc) + timedelta(days=90)).timestamp())

    # Determine status field names based on workflow type
    if workflow_type == "case_check":
        status_field = "caseCheckStatus"
        updated_field = "caseCheckUpdatedAt"
    else:
        status_field = "status"  # Summary workflow (default/legacy)
        updated_field = "updatedAt"

    # Base SET expressions
    set_parts = [f"#{status_field} = :{status_field}", f"{updated_field} = :u", "expiresAt = :e"]
    names = {f"#{status_field}": status_field}
    vals = {f":{status_field}": status_up, ":u": now_iso, ":e": exp_ts}

    # Add metadata fields
    if metadata:
        metadata = _to_ddb_numbers(metadata)
        for k, v in metadata.items():
            # Metadata keys are already workflow-specific (e.g., caseCheckKey vs summaryKey)
            # No need for additional prefixing
            names[f"#{k}"] = k
            vals[f":{k}"] = v
            set_parts.append(f"#{k} = :{k}")

    update_expr = "SET " + ", ".join(set_parts)

    # Update with appropriate condition
    if status_up == "COMPLETED" or force:
        # Always allow COMPLETED or force to write
        JOB_TABLE_REF.update_item(
            Key={"meetingId": meeting_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=vals,
        )
    else:
        # Don't clobber COMPLETED status for THIS workflow type
        vals[":done"] = "COMPLETED"
        JOB_TABLE_REF.update_item(
            Key={"meetingId": meeting_id},
            UpdateExpression=update_expr,
            ConditionExpression=f"attribute_not_exists(#{status_field}) OR #{status_field} <> :done",
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=vals,
        )

    return {
        "success": True,
        "status": status_up
    }
