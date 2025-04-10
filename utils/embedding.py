import os
import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the embedding model
_embedding_model = None


class GoogleGenerativeEmbeddings:
    """
    A wrapper class for Google's GenerativeAI embeddings that provides 
    a LangChain-compatible interface.
    """
    
    def __init__(self, api_key=None, model_name="models/embedding-001"):
        """
        Initialize the Google GenerativeAI embeddings wrapper.
        
        Args:
            api_key: The Google API Key
            model_name: The model name for embeddings (must start with 'models/' or 'tunedModels/')
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("Google API Key is required but not provided")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
    
    def embed_query(self, text):
        """
        Generate embeddings for a single piece of text.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embeddings
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            
            # Return the embedding values
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding for query: {str(e)}")
            raise
    
    def embed_documents(self, documents):
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: A list of text documents to embed
            
        Returns:
            A list of list of floats representing the embeddings for each document
        """
        try:
            embeddings = []
            for doc in documents:
                result = genai.embed_content(
                    model=self.model_name,
                    content=doc,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings for documents: {str(e)}")
            raise


def get_embedding_model():
    """
    Get the embedding model. Creates a new one if it doesn't exist yet.
    
    Returns:
        An embedding model compatible with LangChain's interface
    """
    global _embedding_model
    
    if _embedding_model is None:
        # Try to use Google's GenerativeAI embeddings if possible
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if google_api_key:
            try:
                _embedding_model = GoogleGenerativeEmbeddings(api_key=google_api_key)
                logger.info("Using Google GenerativeAI for embeddings")
                return _embedding_model
            except Exception as e:
                logger.warning(f"Failed to initialize Google embeddings: {str(e)}")
        
        # Fallback to HuggingFace embeddings
        try:
            # Use a smaller model that works well for embeddings
            _embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
            return _embedding_model
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {str(e)}")
            raise ValueError("Could not initialize any embedding model")
    
    return _embedding_model
