import os
import streamlit as st
import base64
import time
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # Local embeddings
from qdrant_client import QdrantClient
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import httpx  # For custom timeout and retry logic
import threading  # For custom timeout logic

# Load environment variables
load_dotenv()

# Initialize FireCrawl
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
if not firecrawl_api_key:
    st.error("Firecrawl API key is missing. Please add it to the .env file.")
    st.stop()

firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)

# Initialize Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="docs_collection")

# Initialize Ollama with DeepSeek-R1 and increased timeout
llm = Ollama(model="deepseek-r1", request_timeout=300.0)  # Increased timeout to 300 seconds

# Initialize Local Embedding Model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # Local embeddings

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model  # Use local embeddings

# Load documents from uploaded files
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        start_time = time.time()
        file_content = uploaded_file.read()
        if uploaded_file.type == "application/pdf":
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                documents.append(Document(text=text))
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")
        else:
            try:
                text = file_content.decode("utf-8")
            except UnicodeDecodeError:
                text = file_content.decode("utf-8", errors="replace")
            documents.append(Document(text=text))
        end_time = time.time()
        st.write(f"Processed {uploaded_file.name} in {end_time - start_time:.2f} seconds")
    return documents

# Create index
def create_index(documents):
    return VectorStoreIndex.from_documents(documents, storage_context=StorageContext.from_defaults(vector_store=vector_store))

# Custom timeout logic for web search
def web_search_with_timeout(query, timeout=60.0):  # Increased timeout to 60 seconds
    def scrape_url():
        try:
            result = firecrawl_app.scrape_url(query, {"pageOptions": {"onlyMainContent": True}})
            return result.get("markdown", "No relevant information found.")
        except Exception as e:
            return f"Error during web search: {str(e)}"

    # Create a thread to run the scrape_url function
    thread = threading.Thread(target=lambda: result.append(scrape_url()))
    result = []  # To store the result from the thread
    thread.start()

    # Wait for the thread to complete or timeout
    thread.join(timeout=timeout)

    if thread.is_alive():
        # If the thread is still alive after the timeout, it means the function took too long
        return "Error: Web search timed out."
    else:
        # Return the result from the thread
        return result[0] if result else "No result found."

# Web search fallback using FireCrawl with retry logic
def web_search(query):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = web_search_with_timeout(query, timeout=60.0)  # Increased timeout to 60 seconds
            if "Error" in result:
                raise httpx.ReadTimeout(result)
            return result
        except httpx.ReadTimeout:
            if attempt < max_retries - 1:
                st.warning(f"Web search timed out. Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)  # Wait before retrying
            else:
                return "Error: Web search timed out after multiple attempts."
        except Exception as e:
            return f"Error during web search: {str(e)}"

# Display PDF in the sidebar
def display_pdf(file_bytes: bytes, file_name: str):
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit app
def main():
    st.title("RAG with DeepSeek-R1")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Chat history

    if "index" not in st.session_state:
        st.session_state.index = None  # Store the VectorStoreIndex

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Add Your PDF Document")
        uploaded_files = st.file_uploader(
            "Upload your documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

            # Load documents and create index
            documents = load_documents(uploaded_files)
            st.session_state.index = create_index(documents)

            # Display PDF preview
            for file in uploaded_files:
                display_pdf(file.getvalue(), file.name)

    # Render existing conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask a question about your PDF...")

    if prompt:
        # 1. Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Get the response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Try document search first
            if st.session_state.index:
                try:
                    query_engine = st.session_state.index.as_query_engine()
                    document_response = query_engine.query(prompt)
                    if document_response.response.strip() == "Empty Response":
                        st.write("No answer found in documents. Falling back to web search...")
                        web_response = web_search(prompt)
                        full_response = web_response
                    else:
                        full_response = document_response.response
                except httpx.ReadTimeout:
                    st.error("The request to the DeepSeek model timed out. Please try again later.")
                    full_response = "Error: The request timed out."
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    full_response = f"Error: {str(e)}"
            else:
                full_response = "Please upload a document first."

            # Simulate typing effect
            lines = full_response.split('\n')
            for i, line in enumerate(lines):
                full_response += line
                if i < len(lines) - 1:  # Don't add newline to the last line
                    full_response += '\n'
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.15)  # Adjust the speed as needed

            # Show the final response without the cursor
            message_placeholder.markdown(full_response)

        # 3. Save assistant's message to session
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()