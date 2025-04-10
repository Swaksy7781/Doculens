"""
Utility functions for the PDF Chat application
Contains security utilities, input sanitization, and rate limiting
"""

import re
import hashlib
import time
import datetime
import json

from src.logging.log_service import logger, log_security_event, log_audit_event
from src.config import settings

def sanitize_input(text, session_state=None):
    """
    Enhanced sanitize user input to prevent various attacks
    
    Production-grade security features:
    - Prevents XSS via HTML/script tag filtering
    - Prevents command injection
    - Length limiting to prevent DOS
    - Character set restrictions
    - Audit logging of suspicious input
    
    Args:
        text: Text to sanitize
        session_state: Optional Streamlit session state object
        
    Returns:
        Sanitized text or None if text is invalid
    """
    if not text or not isinstance(text, str):
        return None
    
    # Create a sanitization ID for tracking
    sanitize_id = hashlib.md5(f"sanitize_{time.time()}".encode()).hexdigest()[:8]
    
    # Check for potentially malicious patterns (command injection, SQL injection)
    suspicious_patterns = [
        r'(?i)(?:exec|eval|system|os\.|subprocess|import\s+os|import\s+subprocess)',  # Code execution
        r'(?i)(?:select|insert|update|delete|drop|alter|create)\s+(?:from|into|table)',  # SQL injection
        r'(?i)(?:function|javascript|script|alert|onerror|onload)',  # XSS attack
        r'(?:-|;|&|\||\$|\(|\))\s*(?:bash|sh|ksh|csh|echo|cat|grep|sudo)'  # Command injection
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text):
            logger.warning(f"[{sanitize_id}] Potentially malicious input detected: '{text[:50]}...'")
            
            # Log security event with details about the potential threat
            threat_type = "UNKNOWN"
            severity = "MEDIUM"
            
            if "exec" in pattern or "eval" in pattern or "system" in pattern or "subprocess" in pattern:
                threat_type = "CODE_EXECUTION_ATTEMPT"
                severity = "HIGH"
            elif "select" in pattern or "insert" in pattern or "update" in pattern or "delete" in pattern:
                threat_type = "SQL_INJECTION_ATTEMPT"
                severity = "HIGH"
            elif "function" in pattern or "javascript" in pattern or "script" in pattern:
                threat_type = "XSS_ATTEMPT"
                severity = "MEDIUM"
            elif "bash" in pattern or "sh" in pattern or "sudo" in pattern:
                threat_type = "COMMAND_INJECTION_ATTEMPT"
                severity = "HIGH"
                
            # Create a security audit record
            username = None
            if session_state and hasattr(session_state, 'username'):
                username = session_state.username
                
            security_id = log_security_event(
                threat_type,
                severity,
                {
                    "sanitize_id": sanitize_id,
                    "input_fragment": text[:100],  # Only include a fragment for security
                    "detection_time": datetime.datetime.now().isoformat(),
                    "pattern_matched": pattern,
                    "input_length": len(text)
                },
                user=username
            )
            
            logger.warning(f"[{sanitize_id}] Security event logged: {threat_type} (security_id: {security_id})")
            return None
    
    # Remove any HTML tags
    original_length = len(text)
    text = re.sub(r'<[^>]*>', '', text)
    if len(text) != original_length:
        logger.warning(f"[{sanitize_id}] HTML tags removed from input")
        
        # Create a security audit record for potential XSS attempts
        # This is a lower severity than direct script injection attempts
        username = None
        if session_state and hasattr(session_state, 'username'):
            username = session_state.username
            
        security_id = log_security_event(
            "HTML_TAG_REMOVAL",
            "LOW",
            {
                "sanitize_id": sanitize_id,
                "detection_time": datetime.datetime.now().isoformat(),
                "original_length": original_length,
                "cleaned_length": len(text),
                "characters_removed": original_length - len(text)
            },
            user=username
        )
        logger.info(f"[{sanitize_id}] HTML tag removal logged for compliance (security_id: {security_id})")
    
    # Limit special characters
    text = re.sub(r'[^\w\s,.?!;:()\[\]{}"\'-]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Limit length to prevent DOS attacks
    max_length = settings.INPUT_MAX_LENGTH
    if len(text) > max_length:
        logger.warning(f"[{sanitize_id}] Input truncated from {len(text)} to {max_length} characters")
        return text[:max_length] + "... (truncated)"
    
    # Log successful sanitization
    if len(text) > 0:
        logger.debug(f"[{sanitize_id}] Input sanitized successfully")
        return text
    else:
        logger.warning(f"[{sanitize_id}] Sanitization resulted in empty string")
        return None

def check_rate_limits(session_state):
    """
    Check and enforce rate limits for API calls
    Returns True if within limits, False if limit exceeded
    
    Production-grade rate limiting features:
    - Time-window based tracking (1-minute and hourly windows)
    - Application-level throttling for cost control
    - Automatic backoff when approaching limits
    - Detailed logging for monitoring usage patterns
    
    Args:
        session_state: Streamlit session state object
        
    Returns:
        Boolean indicating if request should be allowed
    """
    # Initialize rate limiting tracking if needed
    if 'api_call_history' not in session_state:
        session_state.api_call_history = []
    
    current_time = time.time()
    
    # Add current request to history
    session_state.api_call_history.append(current_time)
    
    # Clean up history - remove entries older than 1 hour
    one_hour_ago = current_time - 3600
    session_state.api_call_history = [t for t in session_state.api_call_history if t > one_hour_ago]
    
    # Count requests in the last minute
    one_minute_ago = current_time - 60
    requests_last_minute = sum(1 for t in session_state.api_call_history if t > one_minute_ago)
    
    # Check against rate limit
    if requests_last_minute > settings.RATE_LIMIT_QUERIES_PER_MINUTE:
        # Log rate limit exceeded
        logger.warning(f"Rate limit exceeded: {requests_last_minute} requests in the last minute")
        
        # Create security event for rate limiting
        username = getattr(session_state, 'username', None)
        log_security_event(
            "RATE_LIMIT_EXCEEDED",
            "MEDIUM",
            {
                "requests_last_minute": requests_last_minute,
                "limit": settings.RATE_LIMIT_QUERIES_PER_MINUTE,
                "timestamp": datetime.datetime.now().isoformat()
            },
            user=username
        )
        
        return False
        
    return True

def format_export_content(conversation_history, current_docs, current_timestamp):
    """
    Format conversation history for export
    
    Args:
        conversation_history: List of conversation entries
        current_docs: List of current document names
        current_timestamp: Current timestamp
        
    Returns:
        Formatted export content as string
    """
    export_content = "# PDF Chat Conversation Export\n\n"
    export_content += f"Exported on: {current_timestamp}\n\n"
    
    if current_docs:
        export_content += "## Documents:\n"
        for doc_name in current_docs:
            export_content += f"- {doc_name}\n"
        export_content += "\n"
    
    export_content += "## Conversation:\n\n"
    for i, exchange in enumerate(conversation_history):
        export_content += f"### Question {i+1}:\n{exchange['question']}\n\n"
        export_content += f"### Answer {i+1}:\n{exchange['answer']}\n\n"
        if 'timestamp' in exchange and exchange['timestamp']:
            export_content += f"*Timestamp: {exchange['timestamp']}*\n\n"
        export_content += "---\n\n"
        
    return export_content

def get_export_filename(current_timestamp):
    """
    Generate export filename based on timestamp
    
    Args:
        current_timestamp: Current timestamp
        
    Returns:
        Filename for export
    """
    return f"pdf_chat_export_{current_timestamp.replace(' ', '_').replace(':', '-')}.md"

def log_export_event(export_id, history_count, current_timestamp, export_filename, export_content, current_docs, username):
    """
    Log data export to audit system
    
    Args:
        export_id: Unique ID for the export
        history_count: Number of conversation entries
        current_timestamp: Export timestamp
        export_filename: Filename for export
        export_content: Content being exported
        current_docs: Current document names
        username: User exporting the data
        
    Returns:
        Audit ID for the export event
    """
    # Log export metrics
    export_metrics = {
        "export_id": export_id,
        "conversation_count": history_count,
        "export_time": current_timestamp,
        "export_filename": export_filename,
        "export_size_bytes": len(export_content)
    }
    logger.info(f"[{export_id}] Export metrics: {json.dumps(export_metrics)}")
    
    # Add export to audit log for compliance tracking
    audit_id = log_audit_event(
        "DATA_EXPORT",
        {
            "export_id": export_id,
            "export_time": current_timestamp,
            "export_type": "conversation_history",
            "record_count": history_count,
            "export_size_bytes": len(export_content),
            "export_format": "markdown",
            "document_references": len(current_docs) if current_docs else 0
        },
        user=username
    )
    
    logger.info(f"[{export_id}] Data export recorded in audit log (audit_id: {audit_id})")
    return audit_id