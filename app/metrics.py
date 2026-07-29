"""
Structured logging for every agent interaction, so the project can back up
claims like "resolved without escalation" or "average latency" with real
numbers instead of vibes.
"""
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "interactions.jsonl")


@contextmanager
def track_interaction(user_query: str):
    """Wrap one agent turn; records latency, tools used, and whether it
    ended in an escalation (ticket) vs. a direct resolution.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": user_query,
        "tool_calls": [],
        "escalated": False,
    }
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["latency_seconds"] = round(time.perf_counter() - start, 3)
        _write(record)


def _write(record: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def summarize() -> dict:
    """Quick aggregate stats over logged interactions — useful for a
    README screenshot or an interview talking point.
    """
    if not os.path.exists(LOG_PATH):
        return {"total_interactions": 0}

    total = 0
    escalated = 0
    latencies = []
    tool_usage = {}

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total += 1
            if row.get("escalated"):
                escalated += 1
            latencies.append(row.get("latency_seconds", 0))
            for call in row.get("tool_calls", []):
                tool_usage[call] = tool_usage.get(call, 0) + 1

    return {
        "total_interactions": total,
        "escalation_rate": round(escalated / total, 3) if total else 0,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "tool_usage": tool_usage,
    }
