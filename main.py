# -----------------------------------------------------------------------------
# EngagePro Chatbot - Perplexity Sonar Optimized Version (Part 2)
# -----------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st
import time
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Any, List, Optional

# Import embeddings and your Perplexity LLM from config
from config import hf_embeddings, llm_local as llm 

# --- DATA CONFIG ---
doc_PICKLE = "engagepro.pkl"

# --- UI SETUP & STYLING ---
st.set_page_config(page_title="EngagePro AI Assistant", page_icon="💼", layout="wide")

def apply_custom_ui():
    st.markdown("""
        <style>
        .stChatMessage { border-radius: 12px; }
        .source-container { 
            background-color: #1e1e1e !important; color: #ffffff !important; 
            padding: 15px; border-radius: 8px; border: 1px solid #333333;
            border-left: 6px solid #007bff; margin-top: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        }
        .source-container p, .source-container small, .source-container span, .source-container strong { color: #ffffff !important; }
        .confidence-tag { color: #ffffff !important; font-weight: bold; font-size: 0.85em; 
            background-color: #28a745; padding: 3px 8px; border-radius: 5px; display: inline-block; margin-bottom: 8px; }
        table { color: inherit !important; background-color: transparent !important; }
        </style>
    """, unsafe_allow_html=True)

# --- BYPASS WRAPPER FOR PERPLEXITY ---
class StopWordBypassLLM(BaseChatModel):
    llm: Any 
    def _generate(self, messages: List[Any], stop: Optional[List[str]] = None, **kwargs: Any) -> Any:
        return self.llm._generate(messages, stop=None, **kwargs)
    @property
    def _llm_type(self) -> str: return "perplexity-bypass"

# --- RAG LOGIC ---
def load_knowledge_base():
    if os.path.exists(doc_PICKLE): return pd.read_pickle(doc_PICKLE)
    return pd.DataFrame(columns=["embedding", "source", "content"])

def get_rag_context(query, k=5):
    df = load_knowledge_base()
    if df.empty: return "", []
    q_emb = hf_embeddings.encode([query], convert_to_numpy=True)[0]
    d_embs = np.array(df["embedding"].tolist())
    sims = (d_embs @ q_emb) / (np.linalg.norm(d_embs, axis=1) * np.linalg.norm(q_emb) + 1e-10)
    top_indices = np.argsort(sims)[-k:][::-1]
    results, citations = [], []
    for i in top_indices:
        if sims[i] > 0.28: 
            results.append(df.iloc[i]["content"])
            citations.append({"text": df.iloc[i]["content"], "score": sims[i], "file": df.iloc[i]["source"]})
    return "\n\n".join(results), citations

# --- MAIN APP ---
def main():
    apply_custom_ui()
    df = load_knowledge_base()
    wiki = WikipediaAPIWrapper()

    with st.sidebar:
        st.title("💼 Assistant Panel")
        st.divider()
        mode = st.selectbox("Search Strategy", ["Hybrid (Smart)", "EngagePro Only", "Wikipedia Only"])
        show_sources = st.toggle("Enable Citations", value=True)
        st.metric("EngagePro Chunks", len(df))
        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.session_state.sources = []
            st.rerun()

    st.title("EngagePro Customer Assistant")
    st.info(f"Current Mode: **{mode}**")

    if "messages" not in st.session_state: st.session_state.messages = []
    if "sources" not in st.session_state: st.session_state.sources = []

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_sources and i < len(st.session_state.sources):
                turn_citations = st.session_state.sources[i]
                if turn_citations:
                    with st.expander("🔍 View Sources"):
                        for c in turn_citations:
                            st.markdown(f"""<div class="source-container">
                                <span class="confidence-tag">Match: {c['score']:.1%}</span><br>
                                <p>{c['text']}</p><small>Source: {c['file']}</small></div>""", unsafe_allow_html=True)

    if prompt := st.chat_input("How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status(f"🧠 AI is retrieving {mode} data...", expanded=False) as status:
                
                context = ""
                turn_citations = []
                wiki_data = ""

                # --- IMPROVED: AUTOMATIC QUERY EXPANSION ---
                # We add 'EngagePro' to the search query to improve retrieval precision
                if mode != "Wikipedia Only":
                    search_query = f"EngagePro {prompt}" 
                    context, turn_citations = get_rag_context(search_query)

                if mode == "Wikipedia Only" or (mode == "Hybrid (Smart)" and not context):
                    wiki_data = wiki.run(prompt)

                # Mode-Specific Prompting with strict formatting
                if mode == "EngagePro Only":
                    sys_msg = """You are EngagePro Assistant. 
                    - Use ONLY the 'Company Context' provided. 
                    - Always use **bullet points** and **bold headings** for lists like Core Values.
                    - If data is missing from context, say you cannot find it."""
                else:
                    sys_msg = f"You are EngagePro Assistant in {mode} mode. Provide structured, accurate answers."

                prompt_content = f"Company Context:\n{context}\n\nWikipedia Data:\n{wiki_data}\n\nQuestion: {prompt}"
                
                wrapped_llm = StopWordBypassLLM(llm=llm)
                full_result = wrapped_llm.invoke([
                    SystemMessage(content=sys_msg),
                    HumanMessage(content=prompt_content)
                ]).content
                
                status.update(label="✅ Answer retrieved!", state="complete")

            st.markdown(full_result)
            st.session_state.messages.append({"role": "assistant", "content": full_result})
            st.session_state.sources.append(turn_citations)
            st.rerun()

if __name__ == "__main__":
    main()
