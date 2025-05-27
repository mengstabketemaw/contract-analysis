import fitz
import re
import os
import uuid
import shutil
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load embedding model with CPU mode for Streamlit Cloud
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# FAR/DFARS Clause Pattern Extraction
def extract_clauses(text):
    pattern = r"(\d{2}\.\d{3}-\d{1,2})"
    matches = re.findall(pattern, text)
    return list(set(matches))

# Improved Chunking
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Embedding & FAISS Indexing
def embed_chunks(chunks):
    vectors = EMBED_MODEL.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, vectors, chunks

# Top-K Context Retrieval
def query_index(index, vectors, chunks, user_query, top_k=6):
    query_vec = EMBED_MODEL.encode([user_query])
    D, I = index.search(np.array(query_vec), top_k)
    results = [chunks[i] for i in I[0]]
    return results

# LLM Call via OpenAI SDK (v1.x)
def call_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a senior federal contracting officer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0.3
    )
    return response.choices[0].message.content

# Streamlit UI
def main():
    st.title("GovCon Contract Clause Assistant")

    uploaded_file = st.file_uploader("Upload a government contract PDF", type="pdf")

    if uploaded_file is not None:
        session_id = str(uuid.uuid4())
        temp_dir = f"temp_{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        path = os.path.join(temp_dir, "file.pdf")

        with open(path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            text = extract_text_from_pdf(path)
            if not text.strip():
                st.error("Uploaded file is empty or unreadable.")
                return

            st.subheader("Detected FAR/DFARS Clauses")
            clauses = extract_clauses(text)
            st.write(clauses)

            st.subheader("Ask a Question About the Contract")
            user_query = st.text_input("Your question:")

            if user_query:
                st.info("Processing document and generating response...")
                chunks = chunk_text(text)
                index, vectors, raw_chunks = embed_chunks(chunks)
                context = query_index(index, vectors, raw_chunks, user_query, top_k=6)
                prompt = (
                    "You are a senior federal contracting officer explaining complex FAR/DFARS clauses.\n"
                    "Be concise, legally accurate, and helpful to non-lawyers.\n\n"
                    "Extracted contract context:\n"
                    f"{chr(10).join(context)}\n\n"
                    "User question:\n"
                    f"{user_query}\n\n"
                    "Answer:"
                )
                answer = call_openai(prompt)
                st.success(answer)

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
