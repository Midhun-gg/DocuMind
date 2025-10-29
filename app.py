import streamlit as st
import os
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from llm_handler import LLMHandler
import tempfile
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="DocuMind - AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_indexed' not in st.session_state:
    st.session_state.documents_indexed = []

# Initialize components
@st.cache_resource
def get_vector_store_manager():
    return VectorStoreManager()

@st.cache_resource
def get_llm_handler():
    return LLMHandler()

vector_manager = get_vector_store_manager()
llm_handler = get_llm_handler()

# Title and description
st.title("🧠 DocuMind: AI-Powered Knowledge Indexer")
st.markdown("""
Welcome to **DocuMind** - Your intelligent document assistant that helps you extract insights from your documents instantly!
Upload your documents, ask questions, and get accurate answers with source citations.
""")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or DOCX files"
    )
    
    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                doc_processor = DocumentProcessor()
                all_chunks = []
                
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Process document
                        chunks = doc_processor.process_document(tmp_path, uploaded_file.name)
                        all_chunks.extend(chunks)
                        
                        if uploaded_file.name not in st.session_state.documents_indexed:
                            st.session_state.documents_indexed.append(uploaded_file.name)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                
                # Index documents in vector store
                if all_chunks:
                    st.session_state.vector_store = vector_manager.create_vector_store(all_chunks)
                    st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")
                    st.success(f"📊 Indexed {len(all_chunks)} text chunks")
    
    # Display indexed documents
    if st.session_state.documents_indexed:
        st.subheader("📚 Indexed Documents")
        for doc in st.session_state.documents_indexed:
            st.text(f"✓ {doc}")
    
    # Clear data button
    if st.button("🗑️ Clear All Data"):
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.session_state.documents_indexed = []
        vector_manager.clear_vector_store()
        st.success("All data cleared!")
        st.rerun()

# Main chat interface
st.header("💬 Ask Questions")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['document']}")
                        st.markdown(f"*Page: {source.get('page', 'N/A')}*")
                        st.markdown(f"> {source['text'][:300]}...")
                        st.divider()

# Query input
if st.session_state.vector_store is None:
    st.info("👆 Please upload and process documents first using the sidebar.")
else:
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant documents
                relevant_docs = vector_manager.search_similar(query, k=4)
                
                if relevant_docs:
                    # Generate answer
                    answer, sources = llm_handler.generate_answer(query, relevant_docs)
                    
                    st.markdown(answer)
                    
                    # Add assistant message to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Display sources
                    with st.expander("📄 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** {source['document']}")
                            st.markdown(f"*Page: {source.get('page', 'N/A')}*")
                            st.markdown(f"> {source['text'][:300]}...")
                            st.divider()
                else:
                    response = "I couldn't find relevant information in the documents to answer your question."
                    st.markdown(response)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "sources": []
                    })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    <small>DocuMind v1.0 | Built with Streamlit, ChromaDB, and OpenAI</small>
</div>
""", unsafe_allow_html=True)
