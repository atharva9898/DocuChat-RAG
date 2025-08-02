import streamlit as st
import os
import tempfile
from dotenv import load_dotenv 
# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers 
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import HuggingFaceHub

# Load environment variables from .env file (not strictly needed for local LLM, but good practice)
load_dotenv()

# --- Configuration for Local LLM ---
# IMPORTANT: You need to download a GGUF model file and place it in your project directory.
# For example, you can download 'mistral-7b-instruct-v0.2.Q4_K_M.gguf' from Hugging Face:
# https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
# Place this file in the same directory as your 'app.py' (or 'rag_app.py').
LOCAL_LLM_MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf" # Make sure this file exists in your project folder

# Define the HuggingFace embedding model to use (still local)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# API Configuration
SUPPORTED_APIS = {
    "Local Model": "local",
    "OpenAI GPT": "openai", 
    "Anthropic Claude": "anthropic",
    "Hugging Face Hub": "huggingface"
}

# --- Streamlit UI Setup ---
st.set_page_config(page_title="📄 Chat with Your Documents using RAG", layout="centered")
st.title("📄 Chat with Your Documents using RAG")

st.sidebar.header("1. Upload Your Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="file_uploader"
)

# API Selection
st.sidebar.header("2. Choose LLM Provider")
selected_api = st.sidebar.selectbox(
    "Select your preferred LLM:",
    list(SUPPORTED_APIS.keys()),
    key="api_selection"
)

# API Configuration based on selection
api_key = None
model_name = None

if SUPPORTED_APIS[selected_api] == "openai":
    api_key = st.sidebar.text_input(
        "OpenAI API Key:",
        type="password",
        help="Enter your OpenAI API key"
    )
    model_name = st.sidebar.selectbox(
        "OpenAI Model:",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
        key="openai_model"
    )
elif SUPPORTED_APIS[selected_api] == "anthropic":
    api_key = st.sidebar.text_input(
        "Anthropic API Key:",
        type="password", 
        help="Enter your Anthropic API key"
    )
    model_name = st.sidebar.selectbox(
        "Anthropic Model:",
        ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
        key="anthropic_model"
    )
elif SUPPORTED_APIS[selected_api] == "huggingface":
    api_key = st.sidebar.text_input(
        "Hugging Face API Token:",
        type="password",
        help="Enter your Hugging Face API token"
    )
    model_name = st.sidebar.text_input(
        "Model Repository:",
        value="microsoft/DialoGPT-medium",
        help="Enter the Hugging Face model repository (e.g., microsoft/DialoGPT-medium)"
    )

# Button to trigger document processing
process_button = st.sidebar.button("3. Process Documents", key="process_button")

# --- Functions for RAG Pipeline ---

def get_document_chunks(uploaded_files):
    """
    Loads, parses, and splits uploaded documents into chunks.
    Handles temporary file storage and cleanup.
    """
    all_texts = []
    temp_dir = tempfile.mkdtemp() # Create a temporary directory

    for uploaded_file in uploaded_files:
        # Create a temporary file path
        file_path = os.path.join(temp_dir, uploaded_file.name)

        # Write the uploaded file to the temporary path
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = None
        if uploaded_file.name.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
            continue

        try:
            docs = loader.load()
            all_texts.extend(docs)
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
        finally:
            # Ensure temporary file is deleted
            if os.path.exists(file_path):
                os.remove(file_path)

    # Clean up the temporary directory
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

    if not all_texts:
        return []

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Increased chunk size for better context
        chunk_overlap=200, # Increased overlap for better context flow
        length_function=len
    )
    split_docs = splitter.split_documents(all_texts)
    return split_docs

def get_vector_store(text_chunks):
    """
    Creates embeddings for text chunks and builds a FAISS vector store.
    Uses HuggingFaceEmbeddings.
    """
    with st.spinner(f"Loading embedding model: {EMBEDDING_MODEL_NAME}..."):
        # Initialize HuggingFace Embeddings
        # This will download the model if not already present
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    with st.spinner("Creating knowledge base (FAISS index)..."):
        vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store

def get_llm_instance(api_type, api_key=None, model_name=None):
    """
    Creates and returns an LLM instance based on the selected API type.
    """
    if api_type == "local":
        if not os.path.exists(LOCAL_LLM_MODEL_PATH):
            st.error(f"Local LLM model file not found: {LOCAL_LLM_MODEL_PATH}")
            st.markdown("Please download the model from [https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf) and place it in the same directory as your script.")
            return None
        
        with st.spinner(f"Loading local LLM model: {LOCAL_LLM_MODEL_PATH}... (This may take a while the first time)"):
            llm = CTransformers(
                model=LOCAL_LLM_MODEL_PATH,
                model_type="mistral",
                config={'max_new_tokens': 512, 'temperature': 0.7}
            )
        return llm
    
    elif api_type == "openai":
        if not api_key:
            st.error("Please provide your OpenAI API key.")
            return None
        
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                model_name=model_name or "gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=512
            )
            return llm
        except Exception as e:
            st.error(f"Error initializing OpenAI: {e}")
            return None
    
    elif api_type == "anthropic":
        if not api_key:
            st.error("Please provide your Anthropic API key.")
            return None
        
        try:
            llm = ChatAnthropic(
                api_key=api_key,
                model=model_name or "claude-3-haiku-20240307",
                temperature=0.7,
                max_tokens=512
            )
            return llm
        except Exception as e:
            st.error(f"Error initializing Anthropic: {e}")
            return None
    
    elif api_type == "huggingface":
        if not api_key:
            st.error("Please provide your Hugging Face API token.")
            return None
        
        try:
            llm = HuggingFaceHub(
                huggingfacehub_api_token=api_key,
                repo_id=model_name or "microsoft/DialoGPT-medium",
                model_kwargs={"temperature": 0.7, "max_length": 512}
            )
            return llm
        except Exception as e:
            st.error(f"Error initializing Hugging Face Hub: {e}")
            return None
    
    else:
        st.error(f"Unsupported API type: {api_type}")
        return None

def get_qa_chain(vector_store, api_type, api_key=None, model_name=None):
    """
    Initializes the LLM based on selected API and creates the RetrievalQA chain.
    Includes a custom prompt template for better RAG performance.
    """
    llm = get_llm_instance(api_type, api_key, model_name)
    if not llm:
        return None

    # Define a custom prompt template for the RAG chain
    prompt_template = """
    You are an AI assistant specialized in answering questions based on the provided context.
    Answer the question as truthfully as possible using only the provided context.
    If the answer is not found in the context, clearly state "I don't have enough information in the provided documents to answer that."
    Do not invent information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    # Create a PromptTemplate object
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Main Application Logic ---

# Use st.session_state to store processed data and models
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = [] # For chat history

# Process documents when the button is clicked
if process_button and uploaded_files:
    # Check if API key is required and provided
    api_type = SUPPORTED_APIS[selected_api]
    if api_type != "local" and not api_key:
        st.sidebar.error(f"Please provide your {selected_api} API key before processing.")
    else:
        with st.spinner("Processing documents and creating knowledge base..."):
            st.session_state.document_chunks = get_document_chunks(uploaded_files)
            if st.session_state.document_chunks:
                st.session_state.vector_store = get_vector_store(st.session_state.document_chunks)
                st.session_state.qa_chain = get_qa_chain(
                    st.session_state.vector_store, 
                    api_type, 
                    api_key, 
                    model_name
                )
                if st.session_state.qa_chain:
                    st.sidebar.success(f"Successfully processed {len(st.session_state.document_chunks)} chunks using {selected_api}.")
                    st.sidebar.info("You can now ask questions in the chat interface.")
                else:
                    st.sidebar.error(f"Failed to initialize {selected_api}. Please check your configuration.")
                    st.session_state.vector_store = None
                    st.session_state.qa_chain = None
            else:
                st.sidebar.warning("No valid documents were processed.")
                st.session_state.vector_store = None
                st.session_state.qa_chain = None
elif process_button and not uploaded_files:
    st.sidebar.warning("Please upload documents first before processing.")

# Reset processing if API selection changes
if "previous_api" not in st.session_state:
    st.session_state.previous_api = selected_api
elif st.session_state.previous_api != selected_api:
    st.session_state.previous_api = selected_api
    st.session_state.qa_chain = None
    st.sidebar.info("API selection changed. Please process documents again.")

# Legacy processing code (remove this block)
"""
if process_button and uploaded_files:
    with st.spinner("Processing documents and creating knowledge base..."):
        st.session_state.document_chunks = get_document_chunks(uploaded_files)
        if st.session_state.document_chunks:
            st.session_state.vector_store = get_vector_store(st.session_state.document_chunks)
            st.session_state.qa_chain = get_qa_chain(st.session_state.vector_store, "local")
            if st.session_state.qa_chain: # Check if QA chain was successfully created (model loaded)
                st.sidebar.success(f"Successfully processed {len(st.session_state.document_chunks)} chunks and loaded LLM.")
                st.sidebar.info("You can now ask questions in the chat interface.")
            else:
                st.sidebar.error("Failed to load LLM. Please check model path and file.")
                st.session_state.vector_store = None
                st.session_state.qa_chain = None
        else:
            st.sidebar.warning("No valid documents were processed.")
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
elif process_button and not uploaded_files:
    st.sidebar.warning("Please upload documents first before processing.")
"""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input for chat
query = st.chat_input("Ask a question about your documents:")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.qa_chain:
        with st.spinner("Searching and generating response..."):
            try:
                # Use the qa_chain to get the answer
                result = st.session_state.qa_chain({"query": query})
                answer = result["result"]
                source_documents = result.get("source_documents", [])

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if source_documents:
                        st.markdown("---")
                        st.markdown("**Sources:**")
                        for i, doc in enumerate(source_documents):
                            source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                            page_number = doc.metadata.get('page', 'N/A')
                            st.markdown(f"- **Source {i+1}:** `{source_name}` (Page: {page_number})")

            except Exception as e:
                st.error(f"An error occurred during response generation: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error while processing your request."})
                with st.chat_message("assistant"):
                    st.markdown("Sorry, I encountered an error while processing your request.")
    else:
        with st.chat_message("assistant"):
            st.warning("Please upload documents and click 'Process Documents' first.")

