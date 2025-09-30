from io import BytesIO

import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def read_pdf(file):
    # streamlit files have a 'name' attribute
    filename = getattr(file, "name", "uploaded_file.pdf")
    data = file.read()
    reader = PdfReader(BytesIO(data))
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        metadata = {"source": filename, "page": i + 1}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def build_index(docs, chunk_size, chunk_overlap, k):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(docs)
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    total_tokens = sum(len(enc.encode(chunk.page_content)) for chunk in splits)
    print(f"Total chunks: {len(splits)}, total tokens: {total_tokens}")
    vectorstore = FAISS.from_documents(
        splits, OpenAIEmbeddings(model="text-embedding-3-small")
    )
    # create a vectorstore retriever
    # https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


def format_docs(docs):
    # nice, citeable context string
    return "\n\n".join(
        f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}] {d.page_content}"
        for d in docs
    )
