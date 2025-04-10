import streamlit as st
import pandas as pd
import os
import tempfile
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

from db.connection import get_db_connection
conn = get_db_connection()
print(conn)
from db.repository import (get_documents, get_document_by_id, save_document,
                           get_chat_sessions, create_chat_session,
                           add_message_to_session, get_messages_by_session_id,
                           search_document_chunks, get_document_chunks,
                           create_user, get_user_by_username)
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
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        document_title = st.text_input("Document Title (optional)")

        if uploaded_file is not None and st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(
                            delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Extract text from PDF
                    extracted_text = extract_text_from_pdf(tmp_file_path)

                    # Use uploaded filename if no title provided
                    if not document_title:
                        document_title = uploaded_file.name

                    # Process document and save to database
                    document_id = process_document(
                        text=extracted_text,
                        title=document_title,
                        filename=uploaded_file.name,
                        user_id=st.session_state.user_id)

                    # Clean up temp file
                    os.unlink(tmp_file_path)

                    st.session_state.current_document_id = document_id
                    st.session_state.document_loaded = True
                    st.success("Document processed successfully!")

                    # Create a new chat session for this document
                    session_name = f"Chat about {document_title}"
                    chat_session_id = create_chat_session(
                        user_id=st.session_state.user_id,
                        document_id=document_id,
                        name=session_name)
                    st.session_state.current_chat_session_id = chat_session_id
                    st.session_state.messages = []

                    st.rerun()
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    st.error(f"Error processing document: {str(e)}")

        st.divider()

        # Document selection
        st.subheader("Your Documents")
        try:
            documents = get_documents(st.session_state.user_id)
            if documents:
                document_options = {
                    doc["title"]: doc["id"]
                    for doc in documents
                }
                selected_document = st.selectbox("Select a document",
                                                 options=list(
                                                     document_options.keys()))

                if st.button("Load Document"):
                    document_id = document_options[selected_document]
                    st.session_state.current_document_id = document_id

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
                        session_name = f"Chat about {document['title']}"
                        chat_session_id = create_chat_session(
                            user_id=st.session_state.user_id,
                            document_id=document_id,
                            name=session_name)
                        st.session_state.current_chat_session_id = chat_session_id
                        st.session_state.messages = []

                    st.session_state.document_loaded = True
                    st.rerun()
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
        document = get_document_by_id(st.session_state.current_document_id)
        st.subheader(f"Document: {document['title']}")
        st.caption(f"Uploaded on: {document['created_at']}")

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
                with st.spinner("Searching document..."):
                    try:
                        # Get embedding model
                        embedding_model = get_embedding_model()

                        # Generate embedding for the question
                        query_embedding = embedding_model.embed_query(prompt)

                        # Search for relevant chunks
                        relevant_chunks = search_document_chunks(
                            document_id=st.session_state.current_document_id,
                            query_embedding=query_embedding,
                            limit=5)

                        if relevant_chunks:
                            # Construct context from chunks
                            context = "\n\n".join([
                                chunk["content"] for chunk in relevant_chunks
                            ])

                            # Generate response using GenAI
                            import google.generativeai as genai

                            genai.configure(
                                api_key=os.getenv("GOOGLE_API_KEY"))
                            model = genai.GenerativeModel(
                                'gemini-2.0-flash-thinking-exp')

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
