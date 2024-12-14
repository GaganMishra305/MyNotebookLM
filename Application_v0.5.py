# COMPONENTS USED: 
## 1. LLM ->  llama3.2:1b
## 2. Embedding -> nomic-embed-text:latest
## 3. VectorDb -> Chromadb



import streamlit as st
import logging
import os
import tempfile
import shutil
from multiprocessing import Pool, cpu_count
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv('.env')
# Set protobuf environment variable to avoid error messages
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Streamlit page configuration
st.set_page_config(
    page_title="Multi-PDF RAG Playground",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def process_single_pdf(file_upload) -> List[Any]:
    """
    Process a single PDF file and return document chunks.
    
    Args:
        file_upload (st.UploadedFile): Uploaded PDF file

    Returns:
        List of document chunks
    """
    logger = logging.getLogger(__name__)
    
    # Create a temporary directory for each file
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save file to temp directory
        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
        
        # Load and split the PDF
        loader = UnstructuredPDFLoader(path)
        data = loader.load()
        
        # Add filename as metadata to help with source tracking
        for doc in data:
            doc.metadata['source'] = file_upload.name
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Slightly smaller chunk size
            chunk_overlap=200,  # Increased overlap for context
            length_function=len
        )
        chunks = text_splitter.split_documents(data)
        
        logger.info(f"Processed PDF: {file_upload.name}")
        return chunks
    
    except Exception as e:
        logger.error(f"Error processing PDF {file_upload.name}: {e}")
        return []
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def create_vector_db_from_pdfs(file_uploads) -> Chroma:
    """
    Create a vector database from multiple uploaded PDF files.

    Args:
        file_uploads (List[st.UploadedFile]): List of Streamlit file upload objects containing PDFs.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating vector DB from {len(file_uploads)} PDF files")
    
    # Initialize progress bar
    progress_bar = st.progress(0, text="Processing PDFs...")
    
    try:
        # Use multiprocessing to parallelize PDF processing
        with Pool(processes=min(cpu_count(), len(file_uploads))) as pool:
            # Map processing across files
            results = list(pool.imap_unordered(process_single_pdf, file_uploads))
        
        # Flatten the list of chunks
        all_chunks = [chunk for file_chunks in results for chunk in file_chunks]
        
        # Update progress bar
        progress_bar.progress(0.5, text="Creating vector database...")
        
        # Create vector database
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest",
        )
        
        vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name="multiPDFRAG"
        )
        
        # Complete progress
        progress_bar.progress(100, text="PDF processing complete!")
        
        # List processed files for toast notification
        processed_files = [f.name for f in file_uploads]
        st.toast(f"Processed {len(processed_files)} PDFs: {', '.join(processed_files)}")
        
        logger.info("Vector DB created from multiple PDFs")
        return vector_db
    
    except Exception as e:
        st.error(f"Error processing PDFs: {e}")
        logger.error(f"Error in create_vector_db_from_pdfs: {e}")
        return None

# def process_question(question: str, vector_db: Chroma, selected_model: str) -> Dict[str, str]:
#     """
#     Process a user question using the vector database and selected language model.
    
#     Returns a dictionary with context and response.
#     """
#     logger = logging.getLogger(__name__)
#     logger.info(f"Processing question: {question} using model: {selected_model}")
    
#     # Initialize LLM
#     llm = ChatOllama(model=selected_model)
    
#     # Set up advanced retriever with Maximal Marginal Relevance
#     retriever = vector_db.as_retriever(
#         search_type="mmr",  # Maximal Marginal Relevance
#         search_kwargs={
#             "k": 5,  # Number of documents to retrieve
#             "fetch_k": 20,  # Number of documents to fetch before filtering
#             "lambda_mult": 0.5  # Balance between diversity and relevance
#         }
#     )
    
#     # Query prompt template for multiple perspectives (original implementation)
#     QUERY_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""You are an AI language model assistant. Your task is to generate 3
#         different versions of the given user question to retrieve relevant documents from
#         a vector database. By generating multiple perspectives on the user question, your
#         goal is to help the user overcome some of the limitations of the distance-based
#         similarity search. Provide these alternative questions separated by newlines.
#         Original question: {question}""",
#     )

#     # Use MultiQueryRetriever with the existing prompt
#     multi_retriever = MultiQueryRetriever.from_llm(
#         vector_db.as_retriever(), 
#         llm,
#         prompt=QUERY_PROMPT
#     )

#     # Retrieve relevant documents using both methods
#     retrieved_docs_multi = multi_retriever.get_relevant_documents(question)
#     retrieved_docs_mmr = retriever.get_relevant_documents(question)
    
#     # Combine and deduplicate documents
#     all_docs = list(dict.fromkeys(retrieved_docs_multi + retrieved_docs_mmr))[:5]
    
#     # Prepare context with source information
#     context = "\n\n".join([
#         f"[From {doc.metadata['source']}] {doc.page_content}" 
#         for doc in all_docs
#     ])

#     # RAG prompt template (original implementation)
#     template = """Answer the question based on the following context. 
#     If the context doesn't contain enough information, say so.

#     Context:
#     {context}

#     Question: {question}

#     Provide a detailed answer and cite the sources of information."""

#     prompt = ChatPromptTemplate.from_template(template)

#     # Create chain
#     chain = (
#         {"context": lambda x: context, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     response = chain.invoke(question)
    
#     # Create sources list
#     sources = list(set(doc.metadata['source'] for doc in all_docs))

#     logger.info("Question processed and response generated")
#     return {
#         "context": context,
#         "response": response,
#         "sources": sources
#     }
def process_question(question: str, vector_db: Chroma, selected_model: str) -> Dict[str, str]:
    """
    Process a user question using the vector database and selected language model.
    
    Returns a dictionary with context and response.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Initialize LLM
    llm = ChatOllama(model=selected_model)
    
    # Set up advanced retriever with Maximal Marginal Relevance
    retriever = vector_db.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance
        search_kwargs={
            "k": 5,  # Number of documents to retrieve
            "fetch_k": 20,  # Number of documents to fetch before filtering
            "lambda_mult": 0.5  # Balance between diversity and relevance
        }
    )
    
    # Query prompt template for multiple perspectives (original implementation)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 3
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative queries separated by newlines.
        
        Just give the queries and nothing else, also don't number the queries. Just give the queries.
        Original question: {question}""",
    )

    # Use MultiQueryRetriever with the existing prompt
    multi_retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # Retrieve relevant documents using both methods
    retrieved_docs_multi = multi_retriever.get_relevant_documents(question)
    retrieved_docs_mmr = retriever.get_relevant_documents(question)
    
    # Combine and deduplicate documents using page_content as unique identifier
    seen_contents = set()
    all_docs = []
    for doc in retrieved_docs_multi + retrieved_docs_mmr:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            all_docs.append(doc)
    
    # Limit to top 5 documents
    all_docs = all_docs[:5]
    
    # Prepare context with source information
    context = "\n\n".join([
        f"[From {doc.metadata['source']}] {doc.page_content}" 
        for doc in all_docs
    ])

    # RAG prompt template (original implementation)
    template = """Answer the question based on the following context. 
    If the context doesn't contain enough information, say so.

    Context:
    {context}

    Question: {question}

    Provide a detailed answer and cite the sources of information."""

    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    
    # Create sources list
    sources = list(set(doc.metadata['source'] for doc in all_docs))

    logger.info("Question processed and response generated")
    return {
        "context": context,
        "response": response,
        "sources": sources
    }
    
def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.subheader("📚 Multi-PDF RAG Playground", divider="gray", anchor=False)

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "uploaded_pdf_names" not in st.session_state:
        st.session_state["uploaded_pdf_names"] = []

    # Model selection (hardcoded for simplicity)
    selected_model = 'llama3.2:1b'

    # PDF file uploader (multiple files allowed)
    with col1:
        file_uploads = st.file_uploader(
            "Upload PDF files ↓", 
            type="pdf", 
            accept_multiple_files=True,
            key="multi_pdf_uploader"
        )

        # Process PDFs when files are uploaded
        if file_uploads:
            # Get current uploaded file names
            current_pdf_names = [f.name for f in file_uploads]
            
            # Check if the uploaded files are different from the previous ones
            if set(current_pdf_names) != set(st.session_state["uploaded_pdf_names"]):
                # Clear existing vector DB if different files are uploaded
                if st.session_state["vector_db"] is not None:
                    st.session_state["vector_db"].delete_collection()
                
                # Create new vector DB
                with st.spinner("Processing PDFs..."):
                    st.session_state["vector_db"] = create_vector_db_from_pdfs(file_uploads)
                    
                    # Update uploaded PDFs list with names
                    st.session_state["uploaded_pdf_names"] = current_pdf_names

        # Show list of uploaded PDFs
        if st.session_state["uploaded_pdf_names"]:
            st.write("Uploaded PDFs:")
            for pdf_name in st.session_state["uploaded_pdf_names"]:
                st.write(f"- {pdf_name}")

        # Delete collection button
        if st.session_state["vector_db"] is not None:
            if st.button("⚠️ Clear PDF Collection", type="secondary"):
                st.session_state["vector_db"].delete_collection()
                st.session_state["vector_db"] = None
                st.session_state["uploaded_pdf_names"] = []
                st.session_state["messages"] = []
                st.rerun()

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Chat input and processing
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                # Process and display assistant response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            result = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            
                            # Display context in an expandable section
                            with st.expander("Retrieved Context"):
                                st.markdown(result["context"])
                            
                            # Display response
                            st.markdown(result["response"])
                            
                            # Display sources
                            st.markdown("\n**Sources:**\n" + "\n".join(result["sources"]))

                            # Full response for message history
                            full_response = f"{result['response']}\n\n**Retrieved Context:**\n{result['context']}\n\n**Sources:**\n{', '.join(result['sources'])}"
                        else:
                            st.warning("Please upload PDF files first.")

                # Add assistant response to chat history
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": full_response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload PDF files to begin chatting...")

if __name__ == "__main__":
    main()
