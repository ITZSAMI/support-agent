from app.rag.retrieve import search

TOOL_SCHEMA = {
    "name": "search_knowledge_base",
    "description": (
        "Search the shipping/support FAQ knowledge base for information on "
        "delivery times, returns, damaged items, carriers, or address changes. "
        "Use this for any general policy or how-to question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's question, rephrased as a search query.",
            }
        },
        "required": ["query"],
    },
}


def run(query: str) -> dict:
    hits = search(query, n_results=3)
    if not hits:
        return {"found": False, "results": []}
    return {
        "found": True,
        "results": [
            {"excerpt": h["text"], "source": h["source"]} for h in hits
        ],
    }
