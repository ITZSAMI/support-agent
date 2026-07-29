import json
import os
import uuid
from datetime import datetime, timezone

TICKETS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "logs", "tickets.jsonl"
)

TOOL_SCHEMA = {
    "name": "create_support_ticket",
    "description": (
        "Escalate an issue to a human agent by creating a support ticket. "
        "Use this only when the knowledge base and order lookup cannot "
        "resolve the customer's issue, e.g. damaged items, disputes, or "
        "requests the agent has no tool for."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "issue_summary": {
                "type": "string",
                "description": "A concise summary of the customer's issue.",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Urgency of the issue.",
            },
        },
        "required": ["issue_summary", "priority"],
    },
}


def run(issue_summary: str, priority: str) -> dict:
    ticket = {
        "ticket_id": f"TICKET-{uuid.uuid4().hex[:8].upper()}",
        "issue_summary": issue_summary,
        "priority": priority,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(TICKETS_PATH), exist_ok=True)
    with open(TICKETS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ticket) + "\n")

    return ticket
