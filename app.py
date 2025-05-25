import fitz
import re
import os
import shutil
import uuid
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

# Phase 1: PDF Ingestion
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Phase 2: Clause Extraction
def extract_clauses(text):
    pattern = r"(\d{2}\.\d{3}-\d{1,2})[\s\S]{0,1000}"
    matches = re.findall(pattern, text)
    return list(set(matches))

# Phase 3: Vector Store Build
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def build_vector_db(text, db_path="faiss_index"):
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=c) for c in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(db_path)
    return db

# Phase 4: Explanation Prompt
EXPLANATION_PROMPT = PromptTemplate(
    input_variables=["clause"],
    template="Explain the following FAR/DFARS clause in plain English for a new government contractor:\n\n{clause}\n\nExplanation:"
)

def create_qa_chain(db_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(db_path, embeddings)
    retriever = db.as_retriever()
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": EXPLANATION_PROMPT})
    return qa

# Phase 5: Streamlit UI
def main():
    st.title("GovCon Contract Clause Assistant")
    st.write("Upload a government contract and ask about FAR/DFARS clauses.")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    query = st.text_input("Ask a question about a clause:")

    if uploaded_file is not None:
        session_id = str(uuid.uuid4())
        temp_dir = f"temp_session_{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        temp_pdf_path = os.path.join(temp_dir, "uploaded_contract.pdf")

        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        text = extract_text_from_pdf(temp_pdf_path)
        clauses = extract_clauses(text)
        st.write("Detected Clauses:", clauses)

        db = build_vector_db(text, db_path=os.path.join(temp_dir, "faiss_index"))
        qa_chain = create_qa_chain(db_path=os.path.join(temp_dir, "faiss_index"))

        if query:
            answer = qa_chain.run(query)
            st.write("**Answer:**", answer)

        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
