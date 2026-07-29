"""
Quick demo UI for the support agent.

Run with:
    streamlit run frontend/streamlit_app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.agent import handle_message
from app.metrics import summarize

st.set_page_config(page_title="Support Agent Demo", page_icon="📦")
st.title("📦 Support Agent")
st.caption("RAG + tool-calling demo — ask about orders, shipping policy, or report an issue.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("tool_calls"):
            st.caption(f"🔧 tools used: {', '.join(turn['tool_calls'])}")

if prompt := st.chat_input("e.g. 'Where is my order ORD-1001?'"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = handle_message(prompt)
        st.write(result["answer"])
        if result["tool_calls"]:
            st.caption(f"🔧 tools used: {', '.join(result['tool_calls'])}")
    st.session_state.history.append({
        "role": "assistant",
        "content": result["answer"],
        "tool_calls": result["tool_calls"],
    })

with st.sidebar:
    st.subheader("Metrics")
    if st.button("Refresh"):
        st.rerun()
    st.json(summarize())
