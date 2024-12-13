import streamlit as st
import os
import shutil
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv('.env')

class PDFChatbot:
    def __init__(self, model="llama3.1:8b"):
        """Initialize the chatbot with Ollama model and components."""
        self.llm = ChatOllama(model=model)
        self.embeddings = OllamaEmbeddings(model=model)
        self.vectorstore = None
        self.processed_pdfs = []

    def process_pdfs(self, pdf_files):
        """
        Process uploaded PDF files and create a FAISSDB vector store.
        
        Args:
            pdf_files (list): List of uploaded PDF files
        """
        # Create a temporary directory for PDF processing
        os.makedirs("temp_pdfs", exist_ok=True)
        
        # List to store all document chunks
        all_splits = []
        processed_files = []
        
        # Text splitter for creating chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Process each PDF
        for pdf_file in pdf_files:
            # Save the uploaded file temporarily
            temp_path = os.path.join("temp_pdfs", pdf_file.name)
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            # Load and split PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            splits = text_splitter.split_documents(documents)
            all_splits.extend(splits)
            processed_files.append(pdf_file.name)
        
        # Create FAISSDB vector store
        self.vectorstore = FAISS.from_documents(
            documents=all_splits, 
            embedding=self.embeddings
        )
        
        # Update processed PDFs list
        self.processed_pdfs.extend(processed_files)
        
        return len(all_splits), processed_files

    def clear_documents(self):
        """
        Clear all processed documents and reset the vector store.
        """
        # Remove temporary PDF directory
        if os.path.exists("temp_pdfs"):
            shutil.rmtree("temp_pdfs")
        
        # Remove FAISSDB vector store
        if os.path.exists("FAISS_db"):
            shutil.rmtree("FAISS_db")
        
        # Reset instance variables
        self.vectorstore = None
        self.processed_pdfs = []

    def create_rag_chain(self):
        """
        Create a retrieval-augmented generation chain.
        
        Returns:
            A RAG chain for processing queries with context
        """
        # Retriever from vector store
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Return top 3 most relevant chunks
        )
        
        # Retrieve context
        def retrieve_context(query):
            context = retriever.invoke(query)
            return context
        
        # RAG prompt template
        template = """You are a helpful AI assistant for answering questions based on the context of uploaded PDFs. 
        Use only the following pieces of context to answer the question. 
        If the answer is not in the context then provide the answer yourself but mention it clearly.

        Context: {context}

        Question: {question}

        Helpful Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain
        rag_chain = (
            {"context": retrieve_context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain, retriever

def main():
    # Page configuration
    st.set_page_config(page_title="PDF Chat with RAG", page_icon="📄", layout="wide")
    
    # Sidebar for PDF Upload
    st.sidebar.title("📚 PDF Document Upload")
    
    # Initialize chatbot in session state if not exists
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
    
    # Initialize messages if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # PDF Upload in Sidebar
    uploaded_pdfs = st.sidebar.file_uploader(
        "Upload PDF Files", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    # Clear Documents Button
    if st.sidebar.button("🗑️ Clear Uploaded Documents"):
        with st.sidebar.status("Clearing documents...", expanded=True) as status:
            # Clear documents from chatbot
            st.session_state.chatbot.clear_documents()
            
            # Clear chat messages
            st.session_state.messages = []
            
            status.update(label="Documents Cleared Successfully!", state="complete")
        
        # Rerun to update the app state
        st.rerun()
    
    # Process PDFs button
    if uploaded_pdfs:
        with st.sidebar.status("Processing PDFs...", expanded=True) as status:
            doc_count, processed_files = st.session_state.chatbot.process_pdfs(uploaded_pdfs)
            st.sidebar.write(f"Processed {doc_count} document chunks")
            
            # Display processed PDF names
            st.sidebar.subheader("Processed PDFs:")
            for pdf in processed_files:
                st.sidebar.success(f"✅ {pdf}")
            
            status.update(label="PDFs Processed Successfully!", state="complete")
    
    # Main chat area
    st.title("💬 PDF Chat with Local LLaMA 3.1")
    st.write("Chat with your documents using Retrieval-Augmented Generation")
    
    # Disable chat input if no PDFs processed
    chat_disabled = st.session_state.chatbot.vectorstore is None
    
    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            st.chat_message("human").markdown(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").markdown(message.content)
    
    # Chat input
    if prompt := st.chat_input("Enter your message", disabled=chat_disabled):
        # Disable chat if no vectorstore
        if chat_disabled:
            st.warning("Please upload and process PDFs first!")
            return
        
        # Display user message
        st.chat_message("human").markdown(prompt)
        
        # Add user message to history
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        # Create RAG chain and retriever
        rag_chain, retriever = st.session_state.chatbot.create_rag_chain()
        
        # Retrieve context
        retrieved_context = retriever.invoke(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            # Show context details
            with st.expander("Retrieved Context"):
                for i, doc in enumerate(retrieved_context, 1):
                    st.write(f"Context {i}:")
                    st.write(doc.page_content)
                    st.write("---")
            
            # Prepare loading spinner
            with st.spinner('Generating response...'):
                response_placeholder = st.empty()
                full_response = ""
                
                # Stream response
                for chunk in rag_chain.stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                # Final response
                response_placeholder.markdown(full_response)
        
        # Add AI response to history
        st.session_state.messages.append(AIMessage(content=full_response))

if __name__ == "__main__":
    main()