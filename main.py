import os
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain import hub
from utils import read_pdf, build_index, format_docs

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
        index=4,
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
    "Upload one or more PDFs to be indexed in RAM. Then ask questions about their content."
)

uploaded_files = st.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

ss = st.session_state
ss.setdefault("retriever", None)
ss.setdefault("sources", None)
ss.setdefault("prompt", None)
ss.setdefault("llm", None)

if ss.prompt is None:
    ss.prompt = hub.pull("rlm/rag-prompt")

if ss.llm is None:
    ss.llm = ChatOpenAI(model=model_name, temperature=0)

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "sources" not in st.session_state:
    st.session_state.sources = None


if uploaded_files and st.button("Index Documents"):
    with st.spinner("Indexing..."):
        all_docs = []
        for file in uploaded_files:
            try:
                docs = read_pdf(file)
                all_docs.extend(docs)
                st.toast(f"Indexed {getattr(file, 'name', 'file')} ✅")
            except Exception as e:
                st.toast(f"Error processing {getattr(file, 'name', 'file')}: {e}", icon="❌")
    if not all_docs:
        st.warning("No valid documents found in the uploaded files.")
        ss.retriever = None
    else:
        ss.retriever = build_index(all_docs, chunk_size, chunk_overlap, k=int(k))
        ss.sources = sorted(list({d.metadata['source'] for d in all_docs}))
        st.success("Indexing complete! Ask questions below.")
elif ss.retriever:
    st.success("Indexing complete! Ask questions below.")

st.divider()

if ss.retriever:
    question = st.text_input("Type question...")

    if question:
        with st.spinner():
            
            chain = (
                # retrieve context
                {
                    "context": ss.retriever | RunnableLambda(format_docs), 
                    "question": RunnablePassthrough()}
                # build prompt
                | ss.prompt 
                # call LLM
                | ss.llm 
                # parse output
                | StrOutputParser()
            ) 
            with get_openai_callback() as cb:
                answer = chain.invoke(question)
                print(f"Total Tokens: {cb.total_tokens}, Prompt Tokens: {cb.prompt_tokens}, Completion Tokens: {cb.completion_tokens}, Total Cost (USD): ${cb.total_cost:.6f}")
            if not answer:
                st.error("No answer returned from the model.")
            else:
                st.markdown("💡 Answer")
                st.write(answer)