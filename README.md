# RAG Streamlit Document Chatbot

## Project Overview

This project implements a Retrieval Augmented Generation (RAG) application using Streamlit, allowing users to chat with their own uploaded documents (PDF, DOCX, TXT). The application extends the capabilities of a Large Language Model (LLM) by providing it with real-time, relevant information retrieved directly from the user's documents, thereby reducing hallucinations and grounding responses in factual content.

This project was developed as part of an internship task, focusing on demonstrating a clear understanding of the RAG pipeline and its components.

## Features

* **Multi-document Upload:** Supports uploading multiple PDF, DOCX, and TXT files.
* **Document Parsing & Chunking:** Efficiently extracts text from various document types and splits them into manageable chunks.
* **Local Embeddings:** Uses `sentence-transformers` for generating document embeddings locally, avoiding API costs.
* **FAISS Vector Store:** Utilizes FAISS for fast similarity search to retrieve relevant document passages.
* **Local LLM Integration:** Leverages a local, open-source LLM (`Mistral-7B-Instruct-v0.2-GGUF`) via `ctransformers`, ensuring the application is completely free to run without external API keys.
* **Interactive Chat Interface:** Provides a user-friendly Streamlit interface for asking questions and receiving answers.
* **Source Attribution:** Displays the source document and page number for retrieved information.

## Architecture (RAG Pipeline)

The application follows a standard RAG pipeline:

1.  **Document Loading:** User uploads documents (PDF, DOCX, TXT).
2.  **Text Splitting:** Documents are broken into smaller, overlapping chunks.
3.  **Embedding Generation:** Each chunk is converted into a numerical vector (embedding) using a local embedding model.
4.  **Vector Storage:** Embeddings and their corresponding text chunks are stored in a FAISS in-memory vector database.
5.  **Retrieval:** When a user asks a question, the question is embedded, and the vector database is queried to find the most relevant document chunks.
6.  **Augmented Generation:** The user's question and the retrieved relevant chunks are sent to a local LLM, which generates an answer grounded in the provided context.
7.  **Response Display:** The LLM's answer and source documents are displayed in the Streamlit chat interface.

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
      python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Local LLM Model:**
    This project uses `Mistral-7B-Instruct-v0.2.Q4_K_M.gguf` as the local LLM. You **must** download this file and place it in the root directory of this project.
    * Download from Hugging Face: [https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)
    * The file size is approximately 4.3 GB.

5.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser. The first time you process documents, the embedding model will be downloaded, and the local LLM will be loaded, which may take a few minutes.

## Usage

1.  **Upload Documents:** Use the sidebar to upload one or more PDF, DOCX, or TXT files.
2.  **Process Documents:** Click the "Process Documents" button in the sidebar. This will create the knowledge base from your uploaded files.
3.  **Ask Questions:** Once processing is complete, type your questions in the chat input field and press Enter. The RAG system will retrieve relevant information and generate an answer.

## Future Improvements

* **Persistent Vector Store:** Implement a persistent vector database (e.g., ChromaDB) to save the indexed documents across application restarts.
* **Advanced Retrieval:** Explore re-ranking techniques or more sophisticated retrieval algorithms.
* **User Management:** Add user authentication and personalized document collections.
* **Enhanced Error Handling:** Implement more granular error handling and user feedback mechanisms.

---
*Developed by Atharva Chavan

Linkedin - https://www.linkedin.com/in/atharva-chavan-ab891a203/

Email - atharva.chavan9898@gmail.com

```
