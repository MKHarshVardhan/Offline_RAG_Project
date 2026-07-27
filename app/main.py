"""
Offline Multimodal RAG Assistant — Streamlit placeholder UI.
Run with: streamlit run app/main.py
"""

import streamlit as st

st.title("Offline Multimodal RAG Assistant")

query = st.text_input("Enter your question:")

if query:
    st.markdown(
        f"""
        <div style="
            background-color: #f0f2f6;
            border-left: 4px solid #4a90d9;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 1rem;
        ">
            {query}
        </div>
        """,
        unsafe_allow_html=True,
    )
