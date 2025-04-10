import logging
import json
from typing import List, Dict, Any, Optional
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

from db.repository import save_document, save_document_chunk
from utils.embedding import get_embedding_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_document(text: str, title: str, filename: str, user_id: int, 
                     chunk_size: int = 1000, chunk_overlap: int = 200) -> Optional[int]:
    """
    Process a document by:
    1. Saving it to the database
    2. Splitting it into chunks
    3. Generating embeddings for each chunk
    4. Saving chunks and embeddings to the database
    
    Args:
        text: The full text content of the document
        title: The title of the document
        filename: The original filename
        user_id: The ID of the user who uploaded the document
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        The ID of the newly created document
    """
    try:
        # 1. Save document to database
        document_id = save_document(
            title=title,
            filename=filename,
            content=text,
            user_id=user_id
        )
        
        if not document_id:
            logger.error("Failed to save document")
            return None
        
        # 2. Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        # 3. Get embedding model
        embedding_model = get_embedding_model()
        
        # 4. Process each chunk
        for i, chunk_text in enumerate(chunks):
            # Generate embedding for chunk
            try:
                embedding = embedding_model.embed_query(chunk_text)
                
                # Create metadata (can include page numbers or other info in the future)
                metadata = {
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                # Save chunk with embedding
                chunk_id = save_document_chunk(
                    document_id=document_id,
                    content=chunk_text,
                    embedding=embedding,
                    chunk_order=i,
                    metadata=metadata
                )
                
                if not chunk_id:
                    logger.warning(f"Failed to save chunk {i}")
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
        
        logger.info(f"Processed document with ID {document_id}: {len(chunks)} chunks created")
        return document_id
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise
