import os
import tempfile
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

# Ensure the database directory exists
DB_DIR = "./chroma_db"
os.makedirs(DB_DIR, exist_ok=True)

# Streamlit app configuration
st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Cached model initialization
@st.cache_resource
def get_chat_model():
    return ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

@st.cache_resource
def get_embedding_model():
    return NVIDIAEmbeddings(model="NV-Embed-QA")


# PDF Processing Function
def process_pdf(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load and split PDF
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    embedding_model = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embedding_model, 
        persist_directory=DB_DIR
    )
    
    # Clean up temporary file
    os.unlink(tmp_file_path)
    
    return vectorstore

# Streamlit App UI
st.title("📚 PDF Chatbot")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            st.session_state.vectorstore = process_pdf(uploaded_file)
            st.success("PDF processed successfully!")

# Chat Interface
if st.session_state.vectorstore:
    # Initialize LLM and QA Chain
    llm = get_chat_model()
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Return top 3 most relevant chunks
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True
    )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question about the PDF"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating response..."):
                try:
                    # Run QA chain
                    response = qa_chain({"query": prompt})
                    full_response = response['result']
                    
                    # Display response
                    st.markdown(full_response)
                    
                    # Optional: Show source documents
                    with st.expander("Source Documents"):
                        for doc in response['source_documents']:
                            st.write(f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:300]}...")
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a PDF file in the sidebar to start chatting.")

# Optional cleanup function if needed
def clear_vectorstore():
    st.session_state.vectorstore = None
    st.session_state.messages = []
    if os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)
    st.rerun()

# Add a clear button in the sidebar
with st.sidebar:
    if st.button("Clear Uploaded Document"):
        clear_vectorstore()