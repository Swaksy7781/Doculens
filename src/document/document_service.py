"""
Document service for the PDF Chat application
Handles PDF processing, text extraction, chunking, and caching
"""

import os
import time
import hashlib
import json
import datetime

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.logging.log_service import logger, log_audit_event
from src.config import settings

def compute_document_hash(file_content):
    """
    Compute a unique hash for a document to enable caching
    
    Args:
        file_content: The binary content of the file
        
    Returns:
        A unique hash string representing the document
    """
    return hashlib.md5(file_content).hexdigest()

def get_pdf_text(pdf_docs, document_cache=None):
    """
    Extract text from uploaded PDF documents with progress tracking
    Enhanced with caching for better performance and scalability
    
    Args:
        pdf_docs: List of uploaded PDF documents
        document_cache: Optional cache dictionary to store processed documents
        
    Returns:
        Extracted text from all documents
    """
    # Initialize cache if needed
    if document_cache is None:
        document_cache = {}
        
    # Create a batch ID for this extraction
    batch_id = hashlib.md5(f"extraction_{time.time()}".encode()).hexdigest()[:8]
    
    logger.info(f"[{batch_id}] Starting text extraction for {len(pdf_docs)} documents")
    
    # Check maximum file count
    if len(pdf_docs) > settings.MAX_PDF_COUNT:
        logger.warning(f"[{batch_id}] Too many documents uploaded: {len(pdf_docs)}. Maximum is {settings.MAX_PDF_COUNT}")
        return None
        
    text = ""
    docs_processed = 0
    bytes_processed = 0
    cache_hits = 0
    
    for doc in pdf_docs:
        try:
            # Check file size
            file_size_mb = len(doc.getvalue()) / (1024 * 1024)
            if file_size_mb > settings.MAX_PDF_SIZE_MB:
                logger.warning(f"[{batch_id}] Document {doc.name} exceeds maximum size: {file_size_mb:.2f}MB > {settings.MAX_PDF_SIZE_MB}MB")
                continue
                
            # Compute document hash for caching
            doc_content = doc.getvalue()
            doc_hash = compute_document_hash(doc_content)
            
            # Check if document is in cache
            if document_cache and doc_hash in document_cache:
                logger.info(f"[{batch_id}] Cache hit for document: {doc.name} (hash: {doc_hash[:8]})")
                text += document_cache[doc_hash]
                cache_hits += 1
                continue
                
            logger.info(f"[{batch_id}] Processing document: {doc.name} ({file_size_mb:.2f}MB)")
            
            # Process document
            pdf_reader = PdfReader(doc)
            
            # Get total pages for logging
            total_pages = len(pdf_reader.pages)
            logger.info(f"[{batch_id}] Document {doc.name} has {total_pages} pages")
            
            doc_text = ""
            for i, page in enumerate(pdf_reader.pages):
                doc_text += page.extract_text()
                
            # Update metrics
            bytes_processed += len(doc_content)
            docs_processed += 1
            
            # Store in cache if available
            if document_cache is not None:
                document_cache[doc_hash] = doc_text
                
            text += doc_text
            
        except Exception as e:
            logger.error(f"[{batch_id}] Error processing document {doc.name}: {str(e)}")
            
    # Log extraction metrics
    extraction_metrics = {
        "batch_id": batch_id,
        "documents_processed": docs_processed,
        "documents_skipped": len(pdf_docs) - docs_processed,
        "bytes_processed": bytes_processed,
        "cache_hits": cache_hits,
        "text_length": len(text)
    }
    
    logger.info(f"[{batch_id}] Extraction complete: {json.dumps(extraction_metrics)}")
    
    return text

def get_text_chunks(text):
    """
    Split text into manageable chunks for processing with progress indicator
    
    Args:
        text: The text to be split into chunks
        
    Returns:
        List of text chunks
    """
    # Create a chunking ID for tracking
    chunking_id = hashlib.md5(f"chunking_{time.time()}".encode()).hexdigest()[:8]
    
    logger.info(f"[{chunking_id}] Starting text chunking for {len(text)} characters")
    
    try:
        # Create text splitter with settings from config
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Log chunking metrics
        chunking_metrics = {
            "chunking_id": chunking_id,
            "input_length": len(text),
            "chunks_created": len(chunks),
            "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
        }
        
        logger.info(f"[{chunking_id}] Chunking complete: {json.dumps(chunking_metrics)}")
        
        return chunks
    except Exception as e:
        logger.error(f"[{chunking_id}] Error during text chunking: {str(e)}")
        return None

def log_document_upload(processing_id, pdf_docs, username, retention_policy):
    """
    Log document upload to audit system
    
    Args:
        processing_id: Unique ID for the processing batch
        pdf_docs: List of uploaded PDF documents
        username: User who uploaded the documents
        retention_policy: Selected data retention policy
        
    Returns:
        Audit ID for the event
    """
    # Collect document metadata for audit without storing content
    document_metadata = []
    for doc in pdf_docs:
        file_size_mb = len(doc.getvalue()) / (1024 * 1024)
        document_metadata.append({
            "filename": doc.name,
            "size_mb": round(file_size_mb, 2),
            "content_hash": compute_document_hash(doc.getvalue())[:8]  # First 8 chars only for privacy
        })
    
    # Create audit event
    audit_details = {
        "processing_id": processing_id,
        "document_count": len(pdf_docs),
        "upload_time": datetime.datetime.now().isoformat(),
        "document_metadata": document_metadata,
        "retention_policy": retention_policy
    }
    
    # Log to audit system
    audit_id = log_audit_event(
        "DOCUMENT_UPLOAD",
        audit_details,
        user=username
    )
    
    logger.info(f"[{processing_id}] Document upload recorded in audit log (audit_id: {audit_id})")
    return audit_id

def log_processing_metrics(processing_id, metrics, username):
    """
    Log document processing metrics to audit system
    
    Args:
        processing_id: Unique ID for the processing batch
        metrics: Dictionary of processing metrics
        username: User who processed the documents
        
    Returns:
        Audit ID for the event
    """
    # Add timestamp to metrics
    metrics["processing_time"] = datetime.datetime.now().isoformat()
    
    # Log to audit system
    audit_id = log_audit_event(
        "DOCUMENT_PROCESSING",
        metrics,
        user=username
    )
    
    logger.info(f"[{processing_id}] Processing metrics recorded in audit log (audit_id: {audit_id})")
    return audit_id