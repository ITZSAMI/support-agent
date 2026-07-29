"""
The agent orchestration loop: sends the user's message to Claude with the
tool definitions, executes whichever tools Claude decides to call, feeds
results back, and repeats until Claude produces a final text answer.

This is the "agentic" layer that turns plain RAG into something that can
also act (look up an order, file a ticket) — the piece that most student
RAG projects skip.
"""
import os 
from typing import Optional
from anthropic import Anthropic

from app.tools import knowledge_base, order_status, ticket
from app.metrics import track_interaction

MODEL = "claude-sonnet-4-6"

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


def _execute_tool(name: str, tool_input: dict) -> dict:
    impl = TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        return {"error": f"Unknown tool '{name}'"}
    return impl(**tool_input)


def handle_message(user_message: str, client: Optional[Anthropic] = None) -> dict:
    """Run one full agent turn for a single user message.

    Returns a dict with the final answer text and a trace of tool calls,
    so the caller (API route or Streamlit UI) can show its work.
    """
    client = client or Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    with track_interaction(user_message) as record:
        messages = [{"role": "user", "content": user_message}]

        # Cap iterations so a confused agent can't loop forever.
        for _ in range(5):
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

                result = _execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

            messages.append({"role": "user", "content": tool_results})

        return {
            "answer": "Sorry, I wasn't able to resolve this — escalating to a human agent.",
            "tool_calls": record["tool_calls"],
            "escalated": record["escalated"],
        }
