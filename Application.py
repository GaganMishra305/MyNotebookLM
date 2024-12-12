import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import SentenceTransformer, util
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def get_llm():
    return CTransformers(
        model='Models/llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={
            'max_new_tokens': 512,
        }
    )

def prepare_and_split_docs(pdf_directory):
    split_docs = []
    for pdf in pdf_directory:
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())
        
        loader = PyPDFLoader(pdf.name)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1024,
            chunk_overlap=512,
            disallowed_special=(),
            separators=["\n\n", "\n", " ", ".", ","]
        )
        split_docs.extend(splitter.split_documents(documents))
    return split_docs

def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    db = FAISS.from_documents(split_docs, embeddings)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db

def get_conversation_chain(retriever):
    llm = get_llm()
    
    contextualize_q_system_prompt = (
        "Provide a context-aware response based on chat history and the latest query. "
        "Focus on the most relevant information from the documents."
        "I you dont have a relevant response from context then answer it according to yourself"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "As an AI assistant, provide a concise and accurate answer based on the document context. "
        "Aim for clarity and relevance. Limit your response to the most important information. "
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def calculate_similarity_score(answer: str, context_docs: list) -> float:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    context_docs = [doc.page_content for doc in context_docs]
    
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    context_embeddings = model.encode(context_docs, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(answer_embedding, context_embeddings)
    max_score = similarities.max().item() 
    return max_score

# Initialize session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main Streamlit App
st.title("Document Chat Assistant")

# Sidebar for file upload
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files", 
    type=["pdf"], 
    accept_multiple_files=True
)

# Process Documents Button
if uploaded_files:
    if st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents..."):
            split_docs = prepare_and_split_docs(uploaded_files)
            vector_db = ingest_into_vectordb(split_docs)
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            st.session_state.conversational_chain = get_conversation_chain(retriever)
            st.sidebar.success("Documents processed successfully!")

# Chat input
user_input = st.text_input("Ask a question about your documents")

# Submit button
if st.button("Send"):
    if user_input and 'conversational_chain' in st.session_state:
        with st.spinner("Generating response..."):
            session_id = "abc123"
            conversational_chain = st.session_state.conversational_chain
            response = conversational_chain.invoke(
                {"input": user_input}, 
                config={"configurable": {"session_id": session_id}}
            )
            
            # Store response with context
            st.session_state.chat_history.append({
                "user": user_input, 
                "bot": response['answer'], 
                "context_docs": response.get('context', [])
            })

# Display chat history
if 'chat_history' in st.session_state:
    for index, message in enumerate(st.session_state.chat_history):
        st.write(f"**You:** {message['user']}")
        st.write(f"**Assistant:** {message['bot']}")

        # Source documents expander
        with st.expander(f"Source Documents (Message {index+1})"):
            for doc in message.get('context_docs', []):
                st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.write(doc.page_content)