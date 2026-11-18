"""
The agent orchestration loop: sends the user's message to Claude with the
tool definitions, executes whichever tools Claude decides to call, feeds
results back, and repeats until Claude produces a final text answer.

This is the "agentic" layer that turns plain RAG into something that can
also act (look up an order, file a ticket) — the piece that most student
RAG projects skip.
"""

import json
import os

from anthropic import Anthropic

from app.metrics import track_interaction
from app.tools import knowledge_base, order_status, ticket

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = """You are a customer support agent for a European parcel
delivery company. Answer questions using the search_knowledge_base tool for
policy/how-to questions, check_order_status for order-specific questions,
and create_support_ticket only when you cannot resolve the issue yourself
(e.g. damaged items, disputes, or anything outside your other tools).

Always cite which FAQ entry you used when you answer from the knowledge
base. Be concise and professional. Do not make up order information or
policies that aren't returned by a tool.
"""

TOOLS = [
    knowledge_base.TOOL_SCHEMA,
    order_status.TOOL_SCHEMA,
    ticket.TOOL_SCHEMA,
]

TOOL_IMPLEMENTATIONS = {
    "search_knowledge_base": lambda **kwargs: knowledge_base.run(**kwargs),
    "check_order_status": lambda **kwargs: order_status.run(**kwargs),
    "create_support_ticket": lambda **kwargs: ticket.run(**kwargs),
}


def _execute_tool(name: str, tool_input: dict) -> tuple[dict, bool]:
    """Run one tool. Returns (result, is_error).

    A failing tool must not crash the whole request: the error goes back to
    the model as a tool result, so it can apologise, retry or escalate.
    """
    impl = TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        return {"error": f"Unknown tool '{name}'"}, True
    try:
        return impl(**tool_input), False
    except Exception as exc:  # noqa: BLE001 - report any tool failure to the model
        return {"error": f"{type(exc).__name__}: {exc}"}, True


def handle_message(user_message: str, client: Anthropic | None = None) -> dict:
    """Run one full agent turn for a single user message.

    Returns a dict with the final answer text and a trace of tool calls,
    so the caller (API route or Streamlit UI) can show its work.
    """
    client = client or Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    with track_interaction(user_message) as record:
        messages = [{"role": "user", "content": user_message}]

        # Cap iterations so a confused agent can't loop forever.
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                final_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                return {
                    "answer": final_text,
                    "tool_calls": record["tool_calls"],
                    "escalated": record["escalated"],
                }

            # Assistant turn included at least one tool_use block.
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                record["tool_calls"].append(block.name)
                if block.name == "create_support_ticket":
                    record["escalated"] = True

                result, is_error = _execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                        "is_error": is_error,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        # Out of rounds: actually escalate, so the answer and the metrics
        # both match what happened.
        new_ticket = ticket.run(
            issue_summary=f"Agent could not resolve automatically: {user_message}",
            priority="medium",
        )
        record["tool_calls"].append("create_support_ticket")
        record["escalated"] = True
        return {
            "answer": (
                "Sorry, I wasn't able to resolve this myself. I've passed it to a "
                f"human agent (ticket {new_ticket['ticket_id']})."
            ),
            "tool_calls": record["tool_calls"],
            "escalated": True,
        }
