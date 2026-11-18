import json

from app import agent, metrics
from tests.conftest import FakeAPI, text, tool_use


def last_tool_result(api: FakeAPI) -> dict:
    return api.requests[-1]["messages"][-1]["content"][0]


def test_answers_directly_without_tools():
    api = FakeAPI([text("Hello! How can I help?")])
    result = agent.handle_message("hi", client=api.client())
    assert result == {"answer": "Hello! How can I help?", "tool_calls": [], "escalated": False}


def test_calls_order_tool_and_sends_json_result_back():
    api = FakeAPI(
        [
            tool_use("check_order_status", {"order_id": "ORD-1001"}),
            text("Your order is in transit with DHL."),
        ]
    )
    result = agent.handle_message("Where is ORD-1001?", client=api.client())

    assert result["tool_calls"] == ["check_order_status"]
    assert result["escalated"] is False
    tool_result = last_tool_result(api)
    assert tool_result["is_error"] is False
    assert json.loads(tool_result["content"])["carrier"] == "DHL"  # valid JSON, real data


def test_failing_tool_is_reported_to_the_model_not_raised(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setitem(agent.TOOL_IMPLEMENTATIONS, "search_knowledge_base", broken)
    api = FakeAPI(
        [
            tool_use("search_knowledge_base", {"query": "returns"}),
            text("Sorry, I can't search the FAQ right now."),
        ]
    )
    result = agent.handle_message("How do returns work?", client=api.client())

    assert result["answer"].startswith("Sorry")
    tool_result = last_tool_result(api)
    assert tool_result["is_error"] is True
    assert "vector store unavailable" in tool_result["content"]


def test_unknown_tool_is_an_error_result():
    api = FakeAPI([tool_use("delete_database", {}), text("I can't do that.")])
    agent.handle_message("drop everything", client=api.client())
    assert last_tool_result(api)["is_error"] is True


def test_ticket_marks_interaction_as_escalated():
    api = FakeAPI(
        [
            tool_use(
                "create_support_ticket", {"issue_summary": "Crushed parcel", "priority": "high"}
            ),
            text("I've filed a ticket for you."),
        ]
    )
    result = agent.handle_message("My parcel arrived crushed", client=api.client())
    assert result["escalated"] is True


def test_running_out_of_rounds_really_escalates(temp_logs):
    rounds = [
        tool_use("check_order_status", {"order_id": "ORD-1001"}, f"toolu_{i}")
        for i in range(agent.MAX_TOOL_ROUNDS)
    ]
    api = FakeAPI(rounds)

    result = agent.handle_message("loop forever", client=api.client())

    assert result["escalated"] is True
    assert "TICKET-" in result["answer"]
    tickets = (temp_logs / "tickets.jsonl").read_text().splitlines()
    assert len(tickets) == 1  # a ticket was actually created
    assert metrics.summarize()["escalation_rate"] == 1.0  # and the metrics agree
