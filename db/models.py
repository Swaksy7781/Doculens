from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class User:
    """User model representing application users."""
    id: Optional[int] = None
    username: str = ""
    created_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create a User instance from a dictionary."""
        return cls(
            id=data.get('id'),
            username=data.get('username', ''),
            created_at=data.get('created_at')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert User instance to a dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class Document:
    """Document model representing uploaded PDF documents."""
    id: Optional[int] = None
    title: str = ""
    filename: str = ""
    content: str = ""
    user_id: Optional[int] = None
    created_at: Optional[datetime] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create a Document instance from a dictionary."""
        tags = []
        if 'tags' in data:
            if isinstance(data['tags'], str):
                try:
                    tags = json.loads(data['tags'])
                except:
                    tags = data['tags'].split(',') if data['tags'] else []
            else:
                tags = data['tags'] or []
        
        return cls(
            id=data.get('id'),
            title=data.get('title', ''),
            filename=data.get('filename', ''),
            content=data.get('content', ''),
            user_id=data.get('user_id'),
            created_at=data.get('created_at'),
            tags=tags
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Document instance to a dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'filename': self.filename,
            'content': self.content,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'tags': self.tags
        }


@dataclass
class DocumentChunk:
    """Document chunk model representing a segment of a document with embeddings."""
    id: Optional[int] = None
    document_id: int = 0
    content: str = ""
    embedding: List[float] = None
    chunk_order: int = 0
    chunk_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.embedding is None:
            self.embedding = []
        if self.chunk_metadata is None:
            self.chunk_metadata = {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create a DocumentChunk instance from a dictionary."""
        chunk_metadata = {}
        # Try both field names to ensure compatibility
        metadata_field = 'chunk_metadata' if 'chunk_metadata' in data else 'metadata'
        if metadata_field in data:
            if isinstance(data[metadata_field], str):
                try:
                    chunk_metadata = json.loads(data[metadata_field])
                except:
                    chunk_metadata = {}
            else:
                chunk_metadata = data[metadata_field] or {}
        
        embedding = []
        if 'embedding' in data:
            if isinstance(data['embedding'], str):
                try:
                    embedding = json.loads(data['embedding'])
                except:
                    embedding = []
            else:
                embedding = data['embedding'] or []
        
        return cls(
            id=data.get('id'),
            document_id=data.get('document_id', 0),
            content=data.get('content', ''),
            embedding=embedding,
            chunk_order=data.get('chunk_order', 0),
            chunk_metadata=chunk_metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DocumentChunk instance to a dictionary."""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'content': self.content,
            'embedding': self.embedding,
            'chunk_order': self.chunk_order,
            'chunk_metadata': self.chunk_metadata
        }


@dataclass
class ChatSession:
    """Chat session model representing a conversation about a document."""
    id: Optional[int] = None
    user_id: int = 0
    document_id: int = 0
    name: str = ""
    created_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create a ChatSession instance from a dictionary."""
        return cls(
            id=data.get('id'),
            user_id=data.get('user_id', 0),
            document_id=data.get('document_id', 0),
            name=data.get('name', ''),
            created_at=data.get('created_at')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ChatSession instance to a dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'document_id': self.document_id,
            'name': self.name,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class Message:
    """Message model representing a single message in a chat session."""
    id: Optional[int] = None
    session_id: int = 0
    role: str = ""  # 'user' or 'assistant'
    content: str = ""
    created_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a Message instance from a dictionary."""
        return cls(
            id=data.get('id'),
            session_id=data.get('session_id', 0),
            role=data.get('role', ''),
            content=data.get('content', ''),
            created_at=data.get('created_at')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Message instance to a dictionary."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class Tag:
    """Tag model representing document categories or labels."""
    id: Optional[int] = None
    name: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tag':
        """Create a Tag instance from a dictionary."""
        return cls(
            id=data.get('id'),
            name=data.get('name', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Tag instance to a dictionary."""
        return {
            'id': self.id,
            'name': self.name
        }
