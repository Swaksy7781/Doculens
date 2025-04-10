"""
PDF Chat application with Google's Gemini model
A production-grade document analysis and question answering system

This application allows users to:
1. Upload PDF documents
2. Process and analyze them using LangChain and FAISS vector store
3. Ask questions about the content through a conversational interface
4. Export conversation history

Production features:
- Secure authentication
- Input sanitization 
- Rate limiting
- Comprehensive logging and audit trails
- Document caching
- Data privacy and retention policies
"""

import os
import time
import hashlib
import json
from datetime import datetime

import streamlit as st

# Import application modules
from src.logging.log_service import logger
from src.auth.auth_service import check_auth, log_user_logout
from src.document.document_service import get_pdf_text, get_text_chunks, log_document_upload, log_processing_metrics
from src.vector_store.vector_store_service import get_vector_store, get_conversational_chain, search_documents, get_api_key
from src.utils.helpers import sanitize_input, check_rate_limits, format_export_content, get_export_filename, log_export_event
from src.config import settings

def process_user_input(user_question):
    """
    Process user question and generate response
    
    Args:
        user_question: The question asked by the user
        
    Returns:
        The generated response text or None if processing failed
    """
    # Create a unique ID for this question to track it through the logs
    question_id = hashlib.md5(f"{user_question}:{time.time()}".encode()).hexdigest()[:8]
    
    logger.info(f"[{question_id}] Processing question: {user_question[:50]}...")
    
    # Check rate limits
    if not check_rate_limits(st.session_state):
        st.warning("Rate limit exceeded. Please wait a moment before submitting another question.")
        return None
    
    try:
        # Input validation and sanitization
        logger.debug(f"[{question_id}] Sanitizing user input")
        sanitized_question = sanitize_input(user_question, st.session_state)
        if not sanitized_question:
            message = "Invalid input. Please provide a valid question."
            logger.warning(f"[{question_id}] Invalid input rejected: {user_question[:100]}")
            st.warning(message)
            return None
        
        logger.info(f"[{question_id}] Input sanitized successfully")
            
        # Search for relevant documents
        docs = search_documents(question_id, sanitized_question, st.session_state.username)
        if not docs:
            st.warning("Please upload and process PDF documents first, or try a different question.")
            return None
        
        # Get chain and generate response
        logger.info(f"[{question_id}] Getting conversational chain")
        chain = get_conversational_chain()
        if chain:
            # Generate response
            logger.info(f"[{question_id}] Generating response with LLM")
            llm_start = time.time()
            try:
                response = chain(
                    {"input_documents": docs, "question": sanitized_question},
                    return_only_outputs=True
                )
                
                response_text = response["output_text"]
                llm_time = time.time() - llm_start
                logger.info(f"[{question_id}] Response generated in {llm_time:.2f} seconds")
                
                # Log response metrics
                response_data = {
                    "question_id": question_id,
                    "query": sanitized_question[:100],  # Truncate for log readability
                    "response_time_seconds": llm_time,
                    "document_count": len(docs),
                    "response_length": len(response_text)
                }
                logger.info(f"[{question_id}] Response metrics: {json.dumps(response_data)}")
                
                # Save to conversation history with source info
                source_snippets = []
                for doc in docs:
                    # Get first 100 chars of each source
                    snippet = doc.page_content[:100] + "..."
                    if snippet not in source_snippets:
                        source_snippets.append(snippet)
                
                # Create conversation entry with metadata
                conversation_entry = {
                    "question": user_question,
                    "answer": response_text,
                    "documents": source_snippets,
                    "timestamp": st.session_state.get("current_timestamp", ""),
                    "question_id": question_id,
                    "response_time": llm_time
                }
                
                # Add to session state
                st.session_state.conversation_history.append(conversation_entry)
                logger.info(f"[{question_id}] Added conversation entry to history")
                
                return response_text
            except Exception as e:
                logger.error(f"[{question_id}] Error generating response: {str(e)}")
                st.error(f"Error generating response: {str(e)}")
                return None
        
        return None
    except Exception as e:
        error_message = f"Error processing your question: {str(e)}"
        # Log the full error with traceback
        import traceback
        logger.error(f"[{question_id}] Error processing question: {error_message}")
        logger.error(f"[{question_id}] Exception traceback: {traceback.format_exc()}")
        
        st.error(error_message)
        return None

def main():
    """Main application function"""
    # Log application startup
    startup_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Application starting up at {startup_time}")
    
    # Get system information for diagnostics
    import platform
    import sys
    
    system_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "memory": "Not available"  # Would require psutil which may not be installed
    }
    
    # Log system information on startup
    logger.info(f"System information: {json.dumps(system_info)}")
    
    # Check for API key
    api_key_status = "Available" if get_api_key() else "Missing"
    logger.info(f"Google API key status: {api_key_status}")
    
    # Check for existing vector store
    vector_store_exists = os.path.exists(settings.VECTOR_STORE_PATH)
    logger.info(f"Vector store status: {'Available' if vector_store_exists else 'Not found'}")
    
    # Configure page
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon=settings.APP_ICON,
        layout="wide"
    )
    
    # Initialize authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    if 'username' not in st.session_state:
        st.session_state.username = None
        
    # Authentication block - only show if not authenticated
    if not st.session_state.authenticated:
        st.title(f"{settings.APP_ICON} {settings.APP_TITLE} - Login")
        st.markdown("Please login to access the document analysis system")
        
        # Create login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if check_auth(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.login_time = time.time()
                    logger.info(f"User {username} logged in successfully")
                    st.rerun()  # Refresh the page to show the main app
                else:
                    st.error("Invalid username or password")
                    logger.warning(f"Failed login attempt")
                    
        # Add demo credentials notice (for development only - remove in production)
        st.info("Demo credentials: username=demo, password=demo")
        return  # Exit the function here since we're not authenticated
    
    # If we reach here, user is authenticated, show the main application
    
    # Header with user info
    st.title(f"{settings.APP_ICON} Chat with PDF using Gemini")
    st.markdown(f"""
    Welcome, **{st.session_state.username}**! 
    Upload your PDF documents and ask questions to get detailed answers powered by Google's Gemini 1.5 Pro model.
    """)
    
    # Add logout button in sidebar
    if st.sidebar.button("Logout"):
        # Log the user out
        username = st.session_state.username
        log_user_logout(username, st.session_state.get('login_time', time.time()))
        
        # Clear session state
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.conversation_history = []
        st.rerun()  # Refresh page to show login form
    
    # Initialize session state for conversation history and other variables
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Initialize document caching system
    if 'document_cache' not in st.session_state:
        st.session_state.document_cache = {}
        
    # Initialize rate limiting tracking
    if 'api_call_history' not in st.session_state:
        st.session_state.api_call_history = []
    
    # Initialize performance metrics tracking
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {
            'total_processing_time': 0,
            'total_query_time': 0,
            'document_count': 0,
            'total_chunks': 0,
            'total_queries': 0,
            'avg_query_time': 0
        }
        
    # Set current timestamp for conversations
    st.session_state.current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Sidebar for PDF uploads and settings
    with st.sidebar:
        st.header("📁 Document Upload")
        st.markdown("Upload your PDF files and click 'Process Documents' to analyze them.")
        
        # Add compliance notice
        with st.expander("Data Privacy & Retention"):
            st.markdown("""
            ### Data Privacy Notice
            
            **Document Storage:**
            - Uploaded documents are processed in-memory and their contents are stored temporarily for the current session
            - Document data is not shared with third parties
            - No personal data from documents is permanently stored
            
            **Data Retention:**
            - By default, processed documents are retained only for the current session
            - Conversation history is stored only in your browser session
            - All data is automatically cleared when you close this application
            
            **Your Responsibilities:**
            - Do not upload documents containing sensitive personal data (PII)
            - Ensure you have permission to process any documents you upload
            - Use the export feature to save conversations if needed, as they will not be retained
            """)
        
        # Add settings for data retention (compliance feature)
        with st.expander("Data Retention Settings"):
            selected_retention = st.radio(
                "Document Retention Period:",
                options=settings.RETENTION_OPTIONS,
                index=0
            )
            
            # Store selection in session state
            st.session_state.data_retention_period = selected_retention
            
            # Show explanation based on selection
            if selected_retention == "Session Only (Default)":
                st.info("Documents will be removed when you close the application.")
            elif selected_retention == "1 Hour":
                st.info("Documents will be automatically removed after 1 hour.")
            elif selected_retention == "1 Day":
                st.warning("Documents will be automatically removed after 24 hours. Only select this if necessary.")
            else:
                st.error("Extended retention periods increase security risks. Use only if absolutely necessary.")
        
        # PDF upload with max size warning
        st.warning(f"Maximum file size per PDF: {settings.MAX_PDF_SIZE_MB}MB. Larger files will be rejected.")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            type=["pdf"],
            help="Select one or more PDF files to analyze. Files are processed securely in-memory."
        )
        
        # Process button
        if st.button("Process Documents"):
            if not pdf_docs:
                st.error("Please upload at least one PDF document.")
                logger.warning("User attempted to process documents without uploading any PDFs")
            else:
                # Generate a unique processing ID for tracking this batch
                processing_id = hashlib.md5(f"docs_{time.time()}".encode()).hexdigest()[:8]
                logger.info(f"[{processing_id}] Starting document processing for {len(pdf_docs)} documents")
                
                # Log document upload
                log_document_upload(
                    processing_id, 
                    pdf_docs, 
                    st.session_state.username, 
                    st.session_state.data_retention_period
                )
                
                # Clear conversation history when new documents are processed
                st.session_state.conversation_history = []
                logger.info(f"[{processing_id}] Cleared conversation history")
                
                # Display processing status
                status_container = st.empty()
                status_container.info("Beginning document processing. This may take several minutes for large files...")
                
                # Step 1: Get the text from PDFs
                logger.info(f"[{processing_id}] Starting text extraction")
                extraction_start = time.time()
                raw_text = get_pdf_text(pdf_docs, st.session_state.document_cache)
                extraction_time = time.time() - extraction_start
                
                # Check if text was extracted
                if not raw_text:
                    st.error("Could not extract text from the provided PDFs.")
                    logger.error(f"[{processing_id}] Failed to extract text from PDFs")
                else:
                    # Log extraction metrics
                    text_length = len(raw_text)
                    logger.info(f"[{processing_id}] Text extraction completed in {extraction_time:.2f} seconds. Extracted {text_length} characters")
                    
                    # Step 2: Split the text into chunks
                    status_container.info("PDF text extracted successfully! Now splitting into chunks...")
                    logger.info(f"[{processing_id}] Starting text chunking")
                    chunking_start = time.time()
                    text_chunks = get_text_chunks(raw_text)
                    chunking_time = time.time() - chunking_start
                    
                    # Step 3: Create vector store
                    if text_chunks:
                        chunk_count = len(text_chunks)
                        logger.info(f"[{processing_id}] Text chunking completed in {chunking_time:.2f} seconds. Created {chunk_count} chunks")
                        
                        status_container.info(f"Text splitting complete! Now creating embeddings...")
                        logger.info(f"[{processing_id}] Starting vector store creation")
                        vectorization_start = time.time()
                        success = get_vector_store(text_chunks)
                        vectorization_time = time.time() - vectorization_start
                        
                        if success:
                            total_processing_time = time.time() - extraction_start
                            logger.info(f"[{processing_id}] Vector store created successfully in {vectorization_time:.2f} seconds")
                            logger.info(f"[{processing_id}] Total processing time: {total_processing_time:.2f} seconds")
                            
                            # Log complete processing metrics
                            processing_metrics = {
                                "processing_id": processing_id,
                                "document_count": len(pdf_docs),
                                "text_length": text_length,
                                "chunk_count": chunk_count,
                                "extraction_time_seconds": extraction_time,
                                "chunking_time_seconds": chunking_time,
                                "vectorization_time_seconds": vectorization_time,
                                "total_time_seconds": total_processing_time
                            }
                            
                            # Log metrics to audit system
                            log_processing_metrics(processing_id, processing_metrics, st.session_state.username)
                            
                            status_container.success("✅ Documents processed successfully! You can now ask questions about your document content.")
                            
                            # Save document info in session state
                            doc_names = [doc.name for doc in pdf_docs]
                            st.session_state.current_docs = doc_names
                            logger.info(f"[{processing_id}] Document processing complete. Ready for queries.")
        
        # Add conversation management buttons in columns
        col1, col2 = st.columns(2)
        
        # Clear conversation button
        if col1.button("Clear Conversation"):
            history_count = len(st.session_state.conversation_history)
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")
            logger.info(f"User cleared conversation history. Deleted {history_count} entries.")
            
        # Export conversation button
        if col2.button("Export Conversation"):
            if st.session_state.conversation_history:
                # Generate a unique export ID
                export_id = hashlib.md5(f"export_{time.time()}".encode()).hexdigest()[:8]
                history_count = len(st.session_state.conversation_history)
                logger.info(f"[{export_id}] User requested conversation export with {history_count} entries")
                
                # Generate export content
                export_content = format_export_content(
                    st.session_state.conversation_history,
                    getattr(st.session_state, 'current_docs', []),
                    st.session_state.current_timestamp
                )
                
                # Generate filename
                export_filename = get_export_filename(st.session_state.current_timestamp)
                
                # Provide download link
                st.sidebar.download_button(
                    label="Download Conversation",
                    data=export_content,
                    file_name=export_filename,
                    mime="text/markdown"
                )
                st.sidebar.success("Export ready! Click the download button above.")
                
                # Log export event
                log_export_event(
                    export_id,
                    history_count,
                    st.session_state.current_timestamp,
                    export_filename,
                    export_content,
                    getattr(st.session_state, 'current_docs', []),
                    st.session_state.username
                )
            else:
                logger.warning("User attempted to export empty conversation history")
                st.warning("No conversation history to export.")
        
        # Sidebar information
        st.markdown("---")
        st.markdown("### About")
        st.markdown(f"""
        This application uses:
        - Google's Gemini 1.5 Pro for answering questions
        - FAISS for efficient similarity search
        - LangChain for document processing
        """)
        
        # Add version info and last updated
        st.markdown("---")
        st.caption(f"Version: {settings.APP_VERSION}")
        st.caption(f"Last Updated: {settings.APP_LAST_UPDATED}")
    
    # Main content area
    st.markdown("---")
    
    # Display conversation history
    for i, exchange in enumerate(st.session_state.conversation_history):
        with st.chat_message("user"):
            st.markdown(exchange["question"])
        with st.chat_message("assistant"):
            st.markdown(exchange["answer"])
    
    # Question input using chat input
    user_question = st.chat_input("Ask a question about your documents")
    
    # Process question when submitted
    if user_question:
        # Display user question
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Display assistant response
        with st.chat_message("assistant"):
            # Use a spinner while generating the response
            with st.spinner("Generating response..."):
                # Get response from user_input function
                response = process_user_input(user_question)
                
                # Display the response directly in this chat message
                if response:
                    st.markdown(response)
                    
                    # Add sources with expander
                    with st.expander("View Sources"):
                        # Get the last conversation entry for sources
                        if st.session_state.conversation_history and len(st.session_state.conversation_history) > 0:
                            latest_entry = st.session_state.conversation_history[-1]
                            if "documents" in latest_entry:
                                for i, doc_snippet in enumerate(latest_entry["documents"]):
                                    st.markdown(f"**Source {i+1}**")
                                    st.markdown(f"```\n{doc_snippet}\n```")
                                    st.markdown("---")
    
    # Instructions if no conversation has happened yet
    if not st.session_state.conversation_history and not user_question:
        st.info("👆 Enter your question above to get insights from your documents")
        
        # Example instructions when no documents are processed
        if not os.path.exists(settings.VECTOR_STORE_PATH):
            st.markdown("""
            ### Getting Started:
            1. Upload PDF documents using the sidebar
            2. Click "Process Documents" to analyze them
            3. Ask questions about the content of your documents
            """)

if __name__ == "__main__":
    main()