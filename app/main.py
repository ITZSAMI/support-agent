from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import handle_message
from app.metrics import summarize

app = FastAPI(title="Support Agent API")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[str]
    escalated: bool


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = handle_message(req.message)
    return ChatResponse(**result)


@app.get("/metrics")
def metrics():
    return summarize()


@app.get("/health")
def health():
    return {"status": "ok"}
