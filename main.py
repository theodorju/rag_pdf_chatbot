import os

import streamlit as st
from langchain import hub
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from utils import build_index, format_docs, read_pdf

st.set_page_config(page_title="PDF Q&A with RAG", page_icon="📄", layout="wide")
st.title("📄 PDF Q&A with RAG")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.header("Settings")
    openai_api_key = (
        st.text_input(
            "OpenAI API Key",
            type="password",
            help="Needed only to access OpenAI models.",
        )
        if not OPENAI_API_KEY
        else OPENAI_API_KEY
    )

    model_name = st.selectbox(
        "Model",
        options=[
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "gpt-5-nano",
            "gpt-5-mini",
        ],
        index=3,
        help="Select the model to use for question answering.",
    )

    chunk_size = st.number_input(
        "Chunk Size",
        min_value=500,
        max_value=1000,
        value=1000,
        step=50,
        help="Size of text chunks to split the document into.",
    )
    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Number of overlapping characters between chunks.",
    )
    k = st.number_input(
        "Top-k chunks",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Number of top similar chunks to retrieve for context.",
    )

st.markdown(
    "Upload one or more PDFs to be indexed. Then ask questions about their content."
)

uploaded_files = st.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

state = st.session_state
state.setdefault("retriever", None)
state.setdefault("sources", None)
state.setdefault("prompt", None)
state.setdefault("llm", None)
state.setdefault("chain", None)

if state.prompt is None:
    state.prompt = hub.pull("rlm/rag-prompt")

if state.llm is None:
    state.llm = ChatOpenAI(model=model_name)


if uploaded_files and st.button("Index Documents"):
    with st.spinner("Indexing..."):
        all_docs = []
        for file in uploaded_files:
            try:
                docs = read_pdf(file)
                all_docs.extend(docs)
                st.toast(f"Indexed {getattr(file, 'name', 'file')} ✅")
            except Exception as e:
                st.toast(
                    f"Error processing {getattr(file, 'name', 'file')}: {e}", icon="❌"
                )

    if not all_docs:
        st.warning("No valid documents found in the uploaded files.")
        state.retriever = None

    else:
        state.retriever = build_index(all_docs, chunk_size, chunk_overlap, k=int(k))
        state.sources = sorted(list({d.metadata["source"] for d in all_docs}))
        state.chain = (
            {
                "context": state.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | state.prompt
            | state.llm
            | StrOutputParser()
        )
        st.success("Indexing complete! Ask questions below.")

elif state.retriever:
    st.success("Indexing complete! Ask questions below.")

st.divider()

if state.retriever:
    question = st.text_input("Type question...")

    if question:
        with st.spinner():
            with get_openai_callback() as cb:
                answer = state.chain.invoke(question)
                print(
                    f"Total Tokens: {cb.total_tokens}, Prompt Tokens: {cb.prompt_tokens}, Completion Tokens: {cb.completion_tokens}, Total Cost (USD): ${cb.total_cost:.6f}"
                )
            if not answer:
                st.error("No answer returned from the model.")
            else:
                st.markdown("💡 Answer")
                st.write(answer)
