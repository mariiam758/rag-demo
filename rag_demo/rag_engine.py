from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os
import fitz  # PyMuPDF

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# 📄 Read text from PDFs
def extract_pdf_text(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# 🧠 Smart chunking with sliding window
def split_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# 📥 Load and chunk docs
def load_docs(folder_path):
    all_chunks = []
    sources = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") or filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)

            if filename.endswith(".pdf"):
                text = extract_pdf_text(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

            chunks = split_text(text)
            all_chunks.extend(chunks)
            sources.extend([filename] * len(chunks))

    return all_chunks, sources


# 🏗️ Build FAISS index
def build_faiss_index(docs):
    vectors = embedder.encode(docs)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))
    return index, vectors


# 🔍 Retrieve top-k most relevant chunks
def retrieve(query, docs, sources, index, k=5):
    q_vec = embedder.encode([query])
    D, I = index.search(q_vec, k)
    retrieved_chunks = [docs[i] for i in I[0]]
    retrieved_sources = list(set(sources[i] for i in I[0]))  # remove duplicates
    return retrieved_chunks, retrieved_sources


# ✂️ Trim total context size (optional max_chars control)
def trim_context(chunks, max_chars=3000):
    context = ""
    for chunk in chunks:
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk + "\n\n"
    return context.strip()


# 🤖 Generate answer using LM Studio
def generate_answer(query, context):
    prompt = f"""[INST] Use the following context to answer the question.

Context:
{context}

Question: {query}
[/INST]"""

    try:
        client = openai.OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"
        )

        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct-v0.3",  # <-- use exact ID from LM Studio `/v1/models`
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )

        if not response or not response.choices:
            return "❌ LM Studio returned an empty response."

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ Error during generation:", e)
        return f"❌ Failed to generate response: {e}"
