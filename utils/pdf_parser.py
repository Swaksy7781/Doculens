import logging
import os
import re
import tempfile
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Extracted text from the PDF
    """
    try:
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            text = []
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text.append(page.extract_text())
            
            return "\n\n".join(text)
    except ImportError:
        logger.warning("PyPDF2 not found, trying pdfplumber")
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                text = []
                for page in pdf.pages:
                    text.append(page.extract_text() or "")
                
                return "\n\n".join(text)
        except ImportError:
            logger.error("Neither PyPDF2 nor pdfplumber are available")
            raise ImportError("PDF parsing libraries (PyPDF2, pdfplumber) not found. Please install one of them.")
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise


def get_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Dictionary containing metadata information
    """
    try:
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            info = reader.metadata
            
            # Convert to regular dictionary with string values
            metadata = {}
            if info:
                for key, value in info.items():
                    # Remove the leading slash in keys
                    clean_key = key[1:] if key.startswith('/') else key
                    metadata[clean_key] = str(value)
            
            # Add page count
            metadata['PageCount'] = len(reader.pages)
            
            return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from PDF: {str(e)}")
        return {"error": str(e)}
