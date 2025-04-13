import streamlit as st
import pandas as pd
import os
import tempfile
import time
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

from db.connection import get_db_connection
from db.repository import (get_documents, get_document_by_id, save_document,
                           get_chat_sessions, create_chat_session,
                           add_message_to_session, get_messages_by_session_id,
                           search_document_chunks, search_across_documents, get_document_chunks,
                           create_user, get_user_by_username, get_all_tags, create_tag,
                           add_tag_to_document, remove_tag_from_document)
from utils.document_processor import process_document
from utils.pdf_parser import extract_text_from_pdf
from utils.embedding import get_embedding_model

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize database connection
from db.connection import init_db

init_db()

# Initialize session state variables
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'current_document_id' not in st.session_state:
    st.session_state.current_document_id = None
if 'current_chat_session_id' not in st.session_state:
    st.session_state.current_chat_session_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []
if 'is_batch_session' not in st.session_state:
    st.session_state.is_batch_session = False

# App title and description
st.title("PDF Chat Application")
st.subheader("Ask questions about your PDF documents")

# Sidebar for navigation and document upload
with st.sidebar:
    st.header("Navigation")

    # User login/registration (simplified version)
    username = st.text_input("Username")
    if username and st.button("Login/Register"):
        try:
            # Try to get user, create if doesn't exist
            user = get_user_by_username(username)
            if not user:
                user_id = create_user(username)
                st.success(f"New user registered: {username}")
            else:
                user_id = user["id"]
                st.success(f"Welcome back, {username}!")

            st.session_state.user_id = user_id
            st.rerun()
        except Exception as e:
            st.error(f"Error with authentication: {str(e)}")

    # Only show the rest if user is logged in
    if st.session_state.user_id:
        st.divider()

        # Document upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Choose PDF files",
                                          type="pdf",
                                          accept_multiple_files=True)

        # Display uploaded files
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} file(s):")
            for i, file in enumerate(uploaded_files):
                st.write(f"{i+1}. {file.name}")

        # Option for batch naming or individual titles
        naming_option = st.radio("Document Naming", [
            "Use filenames", "Add a prefix to filenames",
            "Enter custom names later"
        ])

        prefix = ""
        if naming_option == "Add a prefix to filenames":
            prefix = st.text_input("Enter prefix for all documents:")

        # Process button
        if uploaded_files and st.button("Process All Documents"):
            processed_count = 0
            last_document_id = None

            with st.spinner(f"Processing {len(uploaded_files)} documents..."):
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Create a progress bar
                        progress_text = f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})"
                        progress_bar = st.progress(0)

                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(
                                delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        progress_bar.progress(25)

                        # Extract text from PDF
                        extracted_text = extract_text_from_pdf(tmp_file_path)
                        progress_bar.progress(50)

                        # Determine document title based on the selected naming option
                        if naming_option == "Use filenames":
                            document_title = uploaded_file.name
                        elif naming_option == "Add a prefix to filenames":
                            document_title = f"{prefix} {uploaded_file.name}"
                        else:  # Custom names will be added later
                            document_title = uploaded_file.name

                        # Process document and save to database
                        document_id = process_document(
                            text=extracted_text,
                            title=document_title,
                            filename=uploaded_file.name,
                            user_id=st.session_state.user_id)

                        progress_bar.progress(75)

                        # Clean up temp file
                        os.unlink(tmp_file_path)

                        # Keep track of the last document ID for session creation
                        last_document_id = document_id
                        processed_count += 1

                        progress_bar.progress(100)
                        st.success(f"Processed: {document_title}")

                    except Exception as e:
                        logger.error(
                            f"Error processing document {uploaded_file.name}: {str(e)}"
                        )
                        st.error(
                            f"Error processing {uploaded_file.name}: {str(e)}")

            if processed_count > 0:
                st.success(
                    f"Successfully processed {processed_count} out of {len(uploaded_files)} documents!"
                )

                # Set the current document to the last processed one
                if last_document_id is not None:
                    st.session_state.current_document_id = last_document_id
                    st.session_state.document_loaded = True

                    # Get document info for session name
                    document = get_document_by_id(last_document_id)
                    if document is not None:
                        document_title = document.get(
                            'title', f"Document {last_document_id}")

                        # Create a new chat session for the last document
                        session_name = f"Chat about {document_title}"
                        chat_session_id = create_chat_session(
                            user_id=st.session_state.user_id,
                            document_id=last_document_id,
                            name=session_name)
                        st.session_state.current_chat_session_id = chat_session_id
                        st.session_state.messages = []

                        st.info(
                            "Created chat session for the last document. You can select other documents from your library."
                        )
                        time.sleep(2)  # Give user time to read the message
                        st.rerun()
            else:
                st.error("Failed to process any documents. Please try again.")

        st.divider()

        # Document filtering and tagging
        st.subheader("Your Documents")
        try:
            # Get all available tags
            all_tags = get_all_tags()
            tag_names = [tag["name"] for tag in all_tags] if all_tags else []
            
            # Filter documents by tags
            selected_tags = []
            if tag_names:
                with st.expander("Filter Documents by Tags", expanded=False):
                    selected_tags = st.multiselect("Select tags to filter documents", options=tag_names)
            
            # Get documents (filtered by tags if any selected)
            documents = get_documents(st.session_state.user_id)
            
            # Filter documents by selected tags (if any)
            if selected_tags and documents:
                filtered_docs = []
                for doc in documents:
                    doc_tags = doc.get('tags', [])
                    # Convert tags to list if it's a string
                    if isinstance(doc_tags, str):
                        try:
                            doc_tags = json.loads(doc_tags)
                        except:
                            doc_tags = doc_tags.split(',') if doc_tags else []
                    
                    # Check if document has any of the selected tags
                    if any(tag in doc_tags for tag in selected_tags):
                        filtered_docs.append(doc)
                documents = filtered_docs
            
            if documents:
                # Option for single or batch document chat
                chat_mode = st.radio("Chat Mode", ["Single Document", "Multiple Documents (Batch)"], 
                                     help="Select whether to chat with a single document or search across multiple documents")
                
                document_options = {
                    doc["title"]: doc["id"]
                    for doc in documents
                }
                
                if chat_mode == "Single Document":
                    # Reset batch mode if it was set
                    if st.session_state.is_batch_session:
                        st.session_state.is_batch_session = False
                        st.session_state.selected_documents = []
                    
                    selected_document = st.selectbox("Select a document",
                                                   options=list(document_options.keys()),
                                                   key="single_doc_select")

                    if st.button("Load Document"):
                        document_id = document_options[selected_document]
                        st.session_state.current_document_id = document_id
                        st.session_state.is_batch_session = False

                        # Get or create a chat session for this document
                        chat_sessions = get_chat_sessions(
                            user_id=st.session_state.user_id,
                            document_id=document_id)

                        if chat_sessions:
                            # Use the most recent chat session
                            chat_session = chat_sessions[0]
                            st.session_state.current_chat_session_id = chat_session[
                                "id"]

                            # Load messages for this session
                            messages = get_messages_by_session_id(
                                chat_session["id"])
                            st.session_state.messages = messages
                        else:
                            # Create a new chat session
                            document = get_document_by_id(document_id)
                            chat_session_id = None
                            if document is not None:
                                document_title = document.get(
                                    'title', f"Document {document_id}")
                                session_name = f"Chat about {document_title}"
                                chat_session_id = create_chat_session(
                                    user_id=st.session_state.user_id,
                                    document_id=document_id,
                                    name=session_name)

                            if chat_session_id is not None:
                                st.session_state.current_chat_session_id = chat_session_id
                                st.session_state.messages = []

                        st.session_state.document_loaded = True
                        st.rerun()
                else:  # Multiple Documents (Batch)
                    st.markdown("#### Select documents for batch search")
                    
                    # Create checkboxes for document selection
                    selected_doc_ids = []
                    doc_titles = {}
                    
                    for title, doc_id in document_options.items():
                        if st.checkbox(title, key=f"doc_{doc_id}"):
                            selected_doc_ids.append(doc_id)
                            doc_titles[doc_id] = title
                    
                    # Display the selected documents
                    if selected_doc_ids:
                        st.write(f"Selected {len(selected_doc_ids)} documents for batch search")
                        
                        batch_session_name = st.text_input("Name this batch session", 
                                                         value="Multi-Document Research Session")
                        
                        if st.button("Start Batch Chat Session"):
                            # Use the first document as the "anchor" document for the chat session
                            primary_doc_id = selected_doc_ids[0]
                            
                            # Create a new chat session
                            chat_session_id = create_chat_session(
                                user_id=st.session_state.user_id,
                                document_id=primary_doc_id,  # Use the first doc as the primary
                                name=batch_session_name)
                            
                            if chat_session_id is not None:
                                st.session_state.current_chat_session_id = chat_session_id
                                st.session_state.current_document_id = primary_doc_id  # Set primary document
                                st.session_state.selected_documents = selected_doc_ids  # Store all selected docs
                                st.session_state.is_batch_session = True
                                st.session_state.messages = []
                                st.session_state.document_loaded = True
                                
                                st.success(f"Created batch chat session with {len(selected_doc_ids)} documents")
                                st.rerun()
                    else:
                        st.info("Please select at least one document for batch search")
            else:
                st.info("No documents available. Please upload a document.")
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            st.error(f"Error loading documents: {str(e)}")

        st.divider()

        # Chat session selection
        if st.session_state.current_document_id:
            st.subheader("Chat Sessions")
            try:
                chat_sessions = get_chat_sessions(
                    user_id=st.session_state.user_id,
                    document_id=st.session_state.current_document_id)

                if chat_sessions:
                    session_options = {
                        session["name"]: session["id"]
                        for session in chat_sessions
                    }
                    selected_session = st.selectbox(
                        "Select a chat session",
                        options=list(session_options.keys()))

                    if st.button("Load Chat Session"):
                        session_id = session_options[selected_session]
                        st.session_state.current_chat_session_id = session_id

                        # Load messages for this session
                        messages = get_messages_by_session_id(session_id)
                        st.session_state.messages = messages
                        st.rerun()

                # Option to create a new chat session
                new_session_name = st.text_input("New chat session name")
                if new_session_name and st.button("Create New Chat Session"):
                    chat_session_id = create_chat_session(
                        user_id=st.session_state.user_id,
                        document_id=st.session_state.current_document_id,
                        name=new_session_name)
                    st.session_state.current_chat_session_id = chat_session_id
                    st.session_state.messages = []
                    st.rerun()
            except Exception as e:
                logger.error(f"Error with chat sessions: {str(e)}")
                st.error(f"Error with chat sessions: {str(e)}")

# Main content area - Chat interface
if not st.session_state.user_id:
    st.info("Please login or register to use the application.")
elif not st.session_state.document_loaded:
    st.info("Please upload or select a document to chat about.")
else:
    # Display current document info
    try:
        if st.session_state.is_batch_session and st.session_state.selected_documents:
            # Batch session info
            st.subheader("Batch Chat Session")
            st.caption(f"Searching across {len(st.session_state.selected_documents)} documents")
            
            # Optionally display the list of documents in a collapsible section
            with st.expander("Included Documents"):
                for doc_id in st.session_state.selected_documents:
                    doc = get_document_by_id(doc_id)
                    if doc:
                        st.write(f"- {doc.get('title', f'Document {doc_id}')}")
        
        elif st.session_state.current_document_id is not None:
            # Single document info
            document = get_document_by_id(st.session_state.current_document_id)
            if document is not None:
                document_title = document.get('title', 'Document')
                document_date = document.get('created_at', 'Unknown date')
                doc_id = document.get('id')
                
                # Get document tags
                doc_tags = document.get('tags', [])
                # Convert tags to list if it's a string
                if isinstance(doc_tags, str):
                    try:
                        doc_tags = json.loads(doc_tags)
                    except:
                        doc_tags = doc_tags.split(',') if doc_tags else []
                
                # Display document info with tags
                st.subheader(f"Document: {document_title}")
                st.caption(f"Uploaded on: {document_date}")
                
                # Display existing tags
                if doc_tags:
                    st.write("Tags:", ", ".join(doc_tags))
                else:
                    st.write("No tags assigned")
                
                # Tag management in expander
                with st.expander("Manage Tags", expanded=False):
                    # Get all available tags
                    all_tags = get_all_tags()
                    tag_names = [tag["name"] for tag in all_tags] if all_tags else []
                    
                    # Add existing tag
                    if tag_names:
                        tag_to_add = st.selectbox("Add existing tag", options=[""] + tag_names)
                        if tag_to_add and st.button("Add Tag"):
                            if add_tag_to_document(doc_id, tag_to_add):
                                st.success(f"Added tag: {tag_to_add}")
                                st.rerun()
                            else:
                                st.error(f"Failed to add tag: {tag_to_add}")
                    
                    # Create and add new tag
                    new_tag = st.text_input("Create new tag")
                    if new_tag and st.button("Create Tag"):
                        # First create the tag
                        tag_id = create_tag(new_tag)
                        if tag_id:
                            # Then add it to the document
                            if add_tag_to_document(doc_id, new_tag):
                                st.success(f"Created and added tag: {new_tag}")
                                st.rerun()
                            else:
                                st.error(f"Created tag but failed to add it to the document")
                        else:
                            st.error(f"Failed to create tag: {new_tag}")
                    
                    # Remove tag
                    if doc_tags:
                        tag_to_remove = st.selectbox("Remove tag", options=[""] + doc_tags)
                        if tag_to_remove and st.button("Remove Tag"):
                            if remove_tag_from_document(doc_id, tag_to_remove):
                                st.success(f"Removed tag: {tag_to_remove}")
                                st.rerun()
                            else:
                                st.error(f"Failed to remove tag: {tag_to_remove}")

        # Display chat messages
        st.subheader("Chat")
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the document"):
            # Add user message to UI
            with st.chat_message("user"):
                st.write(prompt)

            # Add to session state and save to database
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            if st.session_state.current_chat_session_id:
                add_message_to_session(
                    session_id=st.session_state.current_chat_session_id,
                    role="user",
                    content=prompt)

            # Generate response with citation
            with st.chat_message("assistant"):
                # Adjust spinner text based on batch mode
                spinner_text = "Searching multiple documents..." if st.session_state.is_batch_session else "Searching document..."
                with st.spinner(spinner_text):
                    try:
                        # Get embedding model
                        embedding_model = get_embedding_model()

                        # Generate embedding for the question
                        query_embedding = embedding_model.embed_query(prompt)

                        # Search for relevant chunks based on session mode
                        relevant_chunks = []
                        
                        if st.session_state.is_batch_session and st.session_state.selected_documents:
                            # In batch mode, search across all selected documents
                            relevant_chunks = search_across_documents(
                                user_id=st.session_state.user_id,
                                query_embedding=query_embedding,
                                document_ids=st.session_state.selected_documents,
                                limit=8)  # Increase limit for multi-document search
                        elif st.session_state.current_document_id is not None:
                            # Standard single document search
                            relevant_chunks = search_document_chunks(
                                document_id=st.session_state.current_document_id,
                                query_embedding=query_embedding,
                                limit=5)

                        if relevant_chunks:
                            # Construct context from chunks
                            if st.session_state.is_batch_session:
                                # For batch mode, include document titles with each chunk
                                formatted_chunks = []
                                for chunk in relevant_chunks:
                                    doc_prefix = f"Document: {chunk.get('document_title', 'Unknown')}\n"
                                    formatted_chunks.append(f"{doc_prefix}{chunk['content']}")
                                context = "\n\n".join(formatted_chunks)
                            else:
                                # Standard mode - just the content
                                context = "\n\n".join([
                                    chunk["content"] for chunk in relevant_chunks
                                ])

                            # Generate response using GenAI
                            import google.generativeai as genai

                            genai.configure(
                                api_key=os.getenv("GOOGLE_API_KEY"))
                            model = genai.GenerativeModel(
                                'gemini-2.0-flash-thinking-exp')

                            if st.session_state.is_batch_session:
                                prompt_template = f"""
                                You are an assistant that answers questions about multiple documents.
                                Answer the question based only on the following context from multiple documents:
                                
                                {context}
                                
                                Question: {prompt}
                                
                                If the answer cannot be determined from the context, say "I don't have enough information to answer that question."
                                When providing information, clearly indicate which document the information comes from.
                                Include page numbers or section references if available in the context.
                                """
                            else:
                                prompt_template = f"""
                                You are an assistant that answers questions about documents.
                                Answer the question based only on the following context:
                                
                                {context}
                                
                                Question: {prompt}
                                
                                If the answer cannot be determined from the context, say "I don't have enough information to answer that question."
                                Include page numbers or section references if available in the context.
                                """

                            response = model.generate_content(prompt_template)
                            answer = response.text

                            # Display answer
                            st.write(answer)

                            # Save assistant response
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer
                            })

                            if st.session_state.current_chat_session_id:
                                add_message_to_session(
                                    session_id=st.session_state.
                                    current_chat_session_id,
                                    role="assistant",
                                    content=answer)
                        else:
                            # Customize message based on session mode
                            if st.session_state.is_batch_session:
                                no_info_msg = "I couldn't find relevant information across the selected documents to answer your question. Please try rephrasing your question."
                            else:
                                no_info_msg = "I couldn't find relevant information in the document to answer your question. Please try rephrasing your question."
                            st.write(no_info_msg)

                            # Save assistant response
                            st.session_state.messages.append({
                                "role":
                                "assistant",
                                "content":
                                no_info_msg
                            })

                            if st.session_state.current_chat_session_id:
                                add_message_to_session(
                                    session_id=st.session_state.
                                    current_chat_session_id,
                                    role="assistant",
                                    content=no_info_msg)
                    except Exception as e:
                        logger.error(f"Error generating response: {str(e)}")
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)

                        # Save error message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

                        if st.session_state.current_chat_session_id:
                            add_message_to_session(session_id=st.session_state.
                                                   current_chat_session_id,
                                                   role="assistant",
                                                   content=error_msg)
    except Exception as e:
        logger.error(f"Error in chat interface: {str(e)}")
        st.error(f"Error in chat interface: {str(e)}")
