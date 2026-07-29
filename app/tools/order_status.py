import json
import os

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "mock_orders.json"
)

TOOL_SCHEMA = {
    "name": "check_order_status",
    "description": (
        "Look up the current status, carrier, destination, and estimated "
        "delivery date for a specific order by its order ID."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID, e.g. 'ORD-1001'.",
            }
        },
        "required": ["order_id"],
    },
}


def run(order_id: str) -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        orders = json.load(f)

    order = orders.get(order_id.upper())
    if order is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, "order_id": order_id, **order}
