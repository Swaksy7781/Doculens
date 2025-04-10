import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from psycopg2.sql import SQL, Identifier, Literal
import psycopg2.extras

from db.connection import execute_query
from db.models import User, Document, DocumentChunk, ChatSession, Message, Tag

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# User repository functions
def create_user(username: str) -> int:
    """
    Create a new user in the database.
    
    Args:
        username: The username for the new user
    
    Returns:
        The ID of the newly created user
    """
    query = """
    INSERT INTO users (username, created_at)
    VALUES (%s, NOW())
    RETURNING id
    """
    
    try:
        result = execute_query(query, (username,), fetch_one=True)
        if result and 'id' in result:
            return result['id']
        return None
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a user by their ID.
    
    Args:
        user_id: The ID of the user to retrieve
    
    Returns:
        A dictionary containing user information or None if not found
    """
    query = """
    SELECT id, username, created_at
    FROM users
    WHERE id = %s
    """
    
    try:
        result = execute_query(query, (user_id,), fetch_one=True)
        return result
    except Exception as e:
        logger.error(f"Error getting user by ID: {e}")
        raise


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """
    Get a user by their username.
    
    Args:
        username: The username of the user to retrieve
    
    Returns:
        A dictionary containing user information or None if not found
    """
    query = """
    SELECT id, username, created_at
    FROM users
    WHERE username = %s
    """
    
    try:
        result = execute_query(query, (username,), fetch_one=True)
        return result
    except Exception as e:
        logger.error(f"Error getting user by username: {e}")
        raise


# Document repository functions
def save_document(title: str, filename: str, content: str, user_id: int, tags: List[str] = None) -> int:
    """
    Save a document to the database.
    
    Args:
        title: The title of the document
        filename: The original filename
        content: The full text content of the document
        user_id: The ID of the user who uploaded the document
        tags: Optional list of tags for the document
    
    Returns:
        The ID of the newly created document
    """
    query = """
    INSERT INTO documents (title, filename, content, user_id, created_at, tags)
    VALUES (%s, %s, %s, %s, NOW(), %s)
    RETURNING id
    """
    
    # Convert tags to JSON string
    tags_json = json.dumps(tags) if tags else '[]'
    
    try:
        result = execute_query(query, (title, filename, content, user_id, tags_json), fetch_one=True)
        if result and 'id' in result:
            return result['id']
        return None
    except Exception as e:
        logger.error(f"Error saving document: {e}")
        raise


def get_documents(user_id: int) -> List[Dict[str, Any]]:
    """
    Get all documents for a specific user.
    
    Args:
        user_id: The ID of the user
    
    Returns:
        A list of dictionaries containing document information
    """
    query = """
    SELECT id, title, filename, created_at, tags
    FROM documents
    WHERE user_id = %s
    ORDER BY created_at DESC
    """
    
    try:
        result = execute_query(query, (user_id,), fetch_all=True)
        return result or []
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise


def get_document_by_id(document_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a document by its ID.
    
    Args:
        document_id: The ID of the document to retrieve
    
    Returns:
        A dictionary containing document information or None if not found
    """
    query = """
    SELECT id, title, filename, content, user_id, created_at, tags
    FROM documents
    WHERE id = %s
    """
    
    try:
        result = execute_query(query, (document_id,), fetch_one=True)
        return result
    except Exception as e:
        logger.error(f"Error getting document by ID: {e}")
        raise


def save_document_chunk(document_id: int, content: str, embedding: List[float], 
                         chunk_order: int, metadata: Dict[str, Any] = None) -> int:
    """
    Save a document chunk with its embedding vector to the database.
    
    Args:
        document_id: The ID of the parent document
        content: The text content of this chunk
        embedding: The vector embedding for this chunk
        chunk_order: The position of this chunk in the document
        metadata: Optional metadata for the chunk (page numbers, etc.)
    
    Returns:
        The ID of the newly created document chunk
    """
    # We need to explicitly cast the array to a vector type
    query = """
    INSERT INTO document_chunks (document_id, content, embedding, chunk_order, chunk_metadata)
    VALUES (%s, %s, %s::vector, %s, %s)
    RETURNING id
    """
    
    # Convert metadata to JSON string
    metadata_json = json.dumps(metadata) if metadata else '{}'
    
    try:
        result = execute_query(query, (document_id, content, embedding, chunk_order, metadata_json), fetch_one=True)
        if result and 'id' in result:
            return result['id']
        return None
    except Exception as e:
        logger.error(f"Error saving document chunk: {e}")
        raise


def get_document_chunks(document_id: int) -> List[Dict[str, Any]]:
    """
    Get all chunks for a specific document.
    
    Args:
        document_id: The ID of the document
    
    Returns:
        A list of dictionaries containing chunk information
    """
    query = """
    SELECT id, document_id, content, embedding, chunk_order, chunk_metadata
    FROM document_chunks
    WHERE document_id = %s
    ORDER BY chunk_order ASC
    """
    
    try:
        result = execute_query(query, (document_id,), fetch_all=True)
        return result or []
    except Exception as e:
        logger.error(f"Error getting document chunks: {e}")
        raise


def search_document_chunks(document_id: int, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for document chunks similar to the query embedding.
    
    Args:
        document_id: The ID of the document to search within
        query_embedding: The vector embedding of the query
        limit: Maximum number of results to return
    
    Returns:
        A list of dictionaries containing chunk information ordered by similarity
    """
    # This query uses the pgvector extension to find similar embeddings
    # We need to explicitly cast the array to a vector type
    query = """
    SELECT id, document_id, content, chunk_metadata, 
           1 - (embedding <=> %s::vector) AS similarity
    FROM document_chunks
    WHERE document_id = %s
    ORDER BY similarity DESC
    LIMIT %s
    """
    
    try:
        result = execute_query(query, (query_embedding, document_id, limit), fetch_all=True)
        return result or []
    except Exception as e:
        logger.error(f"Error searching document chunks: {e}")
        raise


# Chat session repository functions
def create_chat_session(user_id: int, document_id: int, name: str) -> int:
    """
    Create a new chat session.
    
    Args:
        user_id: The ID of the user
        document_id: The ID of the document this chat is about
        name: The name of the chat session
    
    Returns:
        The ID of the newly created chat session
    """
    query = """
    INSERT INTO chat_sessions (user_id, document_id, name, created_at)
    VALUES (%s, %s, %s, NOW())
    RETURNING id
    """
    
    try:
        result = execute_query(query, (user_id, document_id, name), fetch_one=True)
        if result and 'id' in result:
            return result['id']
        return None
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise


def get_chat_sessions(user_id: int, document_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get chat sessions for a user, optionally filtered by document.
    
    Args:
        user_id: The ID of the user
        document_id: Optional ID of a specific document
    
    Returns:
        A list of dictionaries containing chat session information
    """
    if document_id:
        query = """
        SELECT cs.id, cs.user_id, cs.document_id, cs.name, cs.created_at,
               d.title as document_title
        FROM chat_sessions cs
        JOIN documents d ON cs.document_id = d.id
        WHERE cs.user_id = %s AND cs.document_id = %s
        ORDER BY cs.created_at DESC
        """
        params = (user_id, document_id)
    else:
        query = """
        SELECT cs.id, cs.user_id, cs.document_id, cs.name, cs.created_at,
               d.title as document_title
        FROM chat_sessions cs
        JOIN documents d ON cs.document_id = d.id
        WHERE cs.user_id = %s
        ORDER BY cs.created_at DESC
        """
        params = (user_id,)
    
    try:
        result = execute_query(query, params, fetch_all=True)
        return result or []
    except Exception as e:
        logger.error(f"Error getting chat sessions: {e}")
        raise


def get_chat_session_by_id(session_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a chat session by its ID.
    
    Args:
        session_id: The ID of the chat session to retrieve
    
    Returns:
        A dictionary containing chat session information or None if not found
    """
    query = """
    SELECT cs.id, cs.user_id, cs.document_id, cs.name, cs.created_at,
           d.title as document_title
    FROM chat_sessions cs
    JOIN documents d ON cs.document_id = d.id
    WHERE cs.id = %s
    """
    
    try:
        result = execute_query(query, (session_id,), fetch_one=True)
        return result
    except Exception as e:
        logger.error(f"Error getting chat session by ID: {e}")
        raise


# Message repository functions
def add_message_to_session(session_id: int, role: str, content: str) -> int:
    """
    Add a message to a chat session.
    
    Args:
        session_id: The ID of the chat session
        role: The role of the message sender (user/assistant)
        content: The content of the message
    
    Returns:
        The ID of the newly created message
    """
    query = """
    INSERT INTO messages (session_id, role, content, created_at)
    VALUES (%s, %s, %s, NOW())
    RETURNING id
    """
    
    try:
        result = execute_query(query, (session_id, role, content), fetch_one=True)
        if result and 'id' in result:
            return result['id']
        return None
    except Exception as e:
        logger.error(f"Error adding message to session: {e}")
        raise


def get_messages_by_session_id(session_id: int) -> List[Dict[str, Any]]:
    """
    Get all messages for a specific chat session.
    
    Args:
        session_id: The ID of the chat session
    
    Returns:
        A list of dictionaries containing message information
    """
    query = """
    SELECT id, session_id, role, content, created_at
    FROM messages
    WHERE session_id = %s
    ORDER BY created_at ASC
    """
    
    try:
        result = execute_query(query, (session_id,), fetch_all=True)
        return result or []
    except Exception as e:
        logger.error(f"Error getting messages by session ID: {e}")
        raise


# Tag repository functions
def create_tag(name: str) -> int:
    """
    Create a new tag.
    
    Args:
        name: The name of the tag
    
    Returns:
        The ID of the newly created tag
    """
    query = """
    INSERT INTO tags (name)
    VALUES (%s)
    ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
    RETURNING id
    """
    
    try:
        result = execute_query(query, (name,), fetch_one=True)
        if result and 'id' in result:
            return result['id']
        return None
    except Exception as e:
        logger.error(f"Error creating tag: {e}")
        raise


def get_all_tags() -> List[Dict[str, Any]]:
    """
    Get all tags.
    
    Returns:
        A list of dictionaries containing tag information
    """
    query = """
    SELECT id, name
    FROM tags
    ORDER BY name ASC
    """
    
    try:
        result = execute_query(query, fetch_all=True)
        return result or []
    except Exception as e:
        logger.error(f"Error getting all tags: {e}")
        raise


def add_tag_to_document(document_id: int, tag_name: str) -> bool:
    """
    Add a tag to a document.
    
    Args:
        document_id: The ID of the document
        tag_name: The name of the tag to add
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # First get the current tags
        document = get_document_by_id(document_id)
        if not document:
            return False
        
        # Parse tags JSON
        current_tags = []
        if document.get('tags'):
            if isinstance(document['tags'], str):
                try:
                    current_tags = json.loads(document['tags'])
                except:
                    current_tags = document['tags'].split(',')
            else:
                current_tags = document['tags']
        
        # Add the new tag if it doesn't exist
        if tag_name not in current_tags:
            current_tags.append(tag_name)
        
        # Update document tags
        query = """
        UPDATE documents
        SET tags = %s
        WHERE id = %s
        """
        
        execute_query(query, (json.dumps(current_tags), document_id))
        return True
    except Exception as e:
        logger.error(f"Error adding tag to document: {e}")
        return False


def remove_tag_from_document(document_id: int, tag_name: str) -> bool:
    """
    Remove a tag from a document.
    
    Args:
        document_id: The ID of the document
        tag_name: The name of the tag to remove
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # First get the current tags
        document = get_document_by_id(document_id)
        if not document:
            return False
        
        # Parse tags JSON
        current_tags = []
        if document.get('tags'):
            if isinstance(document['tags'], str):
                try:
                    current_tags = json.loads(document['tags'])
                except:
                    current_tags = document['tags'].split(',')
            else:
                current_tags = document['tags']
        
        # Remove the tag if it exists
        if tag_name in current_tags:
            current_tags.remove(tag_name)
        
        # Update document tags
        query = """
        UPDATE documents
        SET tags = %s
        WHERE id = %s
        """
        
        execute_query(query, (json.dumps(current_tags), document_id))
        return True
    except Exception as e:
        logger.error(f"Error removing tag from document: {e}")
        return False
