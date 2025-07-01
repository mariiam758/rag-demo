import streamlit as st
import os
from rag_engine import load_docs, build_faiss_index, retrieve, trim_context, generate_answer

# 📁 Setup folders
DOCS_DIR = "docs"
os.makedirs(DOCS_DIR, exist_ok=True)

# 🌐 UI
st.title("📚 RAG Demo with LM Studio")
st.caption("Upload .txt or .pdf files and ask questions based on their content.")

# 📤 File uploader
uploaded_files = st.file_uploader(
    "Drag and drop file here",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("✅ Files uploaded!")

# 🧠 Load and index documents
docs, sources = load_docs(DOCS_DIR)
index, vectors = build_faiss_index(docs)

# 🔍 Question input
query = st.text_input("🔍 Ask a question:")

if query:
    # 🔎 Retrieve top chunks
    retrieved_chunks, used_files = retrieve(query, docs, sources, index, k=5)
    context_text = trim_context(retrieved_chunks, max_chars=3000)

    # 🧠 Generate answer
    answer = generate_answer(query, context_text)

    # 📘 Answer
    st.subheader("📘 Answer")
    st.markdown(answer)

    # 📎 Source files
        # 📎 Source files (numbered)
    if used_files:
        st.subheader("📎 Source Files")
        for i, file in enumerate(used_files, 1):
            st.markdown(f"{i}. `{file}`")

