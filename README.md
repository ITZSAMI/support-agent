# Support Agent — RAG + Tool-Calling Demo

A customer support agent for a fictional cross-border parcel delivery
company. It combines retrieval-augmented generation (RAG) over an FAQ
knowledge base with function-calling tools that let it *act* — look up an
order, or escalate to a human — rather than only answer questions.

## Why this exists

Most student RAG projects stop at "answer questions from documents." This
one adds the agentic layer on top: the model decides, per query, whether to
search the knowledge base, call a tool, or both — and every interaction is
logged so the agent's behavior can be measured, not just demoed.

## Architecture

```
User query
   │
   ▼
Claude (tool-calling loop)
   │
   ├── search_knowledge_base(query)   → RAG retrieval over data/docs/*.md (Chroma)
   ├── check_order_status(order_id)   → mock lookup against data/mock_orders.json
   └── create_support_ticket(...)     → escalation, logged to logs/tickets.jsonl
   │
   ▼
Final answer, with a trace of which tools were used
```

Every turn is wrapped in `app/metrics.py`, which logs latency, which tools
were called, and whether the interaction ended in an escalation. `GET
/metrics` (or the Streamlit sidebar) surfaces aggregate stats: escalation
rate, average latency, tool usage counts.

## Project structure

```
support-agent/
├── app/
│   ├── main.py            # FastAPI app (POST /chat, GET /metrics)
│   ├── agent.py            # orchestration loop: LLM call ↔ tool execution
│   ├── metrics.py          # structured JSONL logging + aggregate stats
│   ├── tools/
│   │   ├── knowledge_base.py
│   │   ├── order_status.py
│   │   └── ticket.py
│   └── rag/
│       ├── ingest.py       # chunks + embeds docs into Chroma
│       └── retrieve.py     # query-time retrieval
├── data/
│   ├── docs/faq.md          # source knowledge base
│   └── mock_orders.json     # mock order database
├── frontend/
│   └── streamlit_app.py     # chat UI + live metrics
└── logs/                    # interactions.jsonl, tickets.jsonl (generated)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then add your ANTHROPIC_API_KEY to .env

# Ingest the knowledge base into the local vector store (run once,
# or again after editing data/docs/*.md)
python -m app.rag.ingest
```

## Running it

**API:**
```bash
uvicorn app.main:app --reload
# POST http://localhost:8000/chat  {"message": "Where is order ORD-1001?"}
```

**Demo UI:**
```bash
streamlit run frontend/streamlit_app.py
```

## Example interactions

> **User:** Where is my order ORD-1001?
> **Agent:** Order ORD-1001 is currently in transit to Berlin, DE via DHL,
> with an estimated delivery of 2026-07-27.
> 🔧 `check_order_status`

> **User:** How long do returns take to process?
> **Agent:** You can start a return from your order history — a prepaid
> label arrives within 24 hours, and packages must be sent back within 14
> days of delivery. *(source: faq.md)*
> 🔧 `search_knowledge_base`

> **User:** My package arrived completely crushed, this is the second time.
> **Agent:** I'm sorry to hear that — I've filed a ticket (priority: high)
> for a human agent to follow up on the damage and the repeat issue.
> 🔧 `search_knowledge_base`, `create_support_ticket`

## Metrics (example, after a handful of test queries)

```json
{
  "total_interactions": 12,
  "escalation_rate": 0.17,
  "avg_latency_seconds": 2.4,
  "tool_usage": {
    "search_knowledge_base": 8,
    "check_order_status": 5,
    "create_support_ticket": 2
  }
}
```

## Possible extensions

- Swap the mock order DB / ticket log for a real database
- Add a `get_shipping_estimate(destination)` tool
- Add automated eval queries to track answer quality over time, not just
  latency/escalation rate
- Multi-turn memory across a conversation (currently each `/chat` call is
  stateless per request; the Streamlit UI keeps history client-side only)
