import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub  # or use OpenAI if available

# -----------------------------
# Configuration
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_resource
def load_llm():
    # For local open-source model via Hugging Face Hub
    return HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5})

@st.cache_resource
def create_vector_db(files):
    documents = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        ext = os.path.splitext(file.name)[-1].lower()

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        else:
            st.warning(f"Unsupported file type: {ext}")
            continue

        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="📚 Document Q&A Chatbot", page_icon="🧠")
st.title("🧠 Chat with Your Documents")

# Upload Files
uploaded_files = st.file_uploader("Upload PDF/DOCX/TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    db = create_vector_db(uploaded_files)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = load_llm()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    st.success("✅ Ready! Ask a question about your documents.")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = qa_chain.invoke({"query": prompt})
        answer = response["result"]
        sources = list(set([doc.metadata['source'] for doc in response.get("source_documents", [])]))

        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                st.caption("📄 Sources:")
                for src in sources:
                    st.caption(f"- {os.path.basename(src)}")

        st.session_state.messages.append({"role": "assistant", "content": answer})