import json

import pytest

from app import metrics
from app.rag.ingest import chunk_markdown, load_documents
from app.tools import order_status, ticket


def test_order_lookup_is_case_insensitive():
    result = order_status.run("ord-1002")
    assert result["found"] is True
    assert result["carrier"] == "Hermes"


def test_unknown_order():
    assert order_status.run("ORD-9999") == {"found": False, "order_id": "ORD-9999"}


def test_ticket_is_written(temp_logs):
    created = ticket.run("Wrong address", "low")
    assert created["ticket_id"].startswith("TICKET-")
    saved = json.loads((temp_logs / "tickets.jsonl").read_text())
    assert saved["issue_summary"] == "Wrong address"


def test_chunking_keeps_one_section_per_heading():
    text = "# FAQ\nIntro\n## Returns\nWithin 14 days.\n## Damage\nSend photos."
    assert chunk_markdown(text) == [
        "# FAQ\nIntro",
        "## Returns\nWithin 14 days.",
        "## Damage\nSend photos.",
    ]


def test_faq_loads_with_unique_ids():
    docs = load_documents()
    assert docs
    assert len({d["id"] for d in docs}) == len(docs)


def test_metrics_summary():
    assert metrics.summarize() == {"total_interactions": 0}
    with metrics.track_interaction("a") as record:
        record["tool_calls"].append("check_order_status")
    with metrics.track_interaction("b") as record:
        record["escalated"] = True
    summary = metrics.summarize()
    assert summary["total_interactions"] == 2
    assert summary["escalation_rate"] == 0.5
    assert summary["tool_usage"] == {"check_order_status": 1}


def test_interaction_is_logged_even_if_it_fails():
    with pytest.raises(ValueError), metrics.track_interaction("boom"):
        raise ValueError("fail")
    assert metrics.summarize()["total_interactions"] == 1
