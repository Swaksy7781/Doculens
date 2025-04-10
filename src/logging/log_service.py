"""
Logging service for the application
Handles application logs, audit logs, and security logs
"""

import os
import logging
import datetime
import uuid
import json

from src.config import settings

# Create logger
logger = logging.getLogger("pdf_chat")
logger.setLevel(logging.INFO)

# Create formatters
app_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
audit_formatter = logging.Formatter('%(asctime)s - AUDIT - %(message)s')
security_formatter = logging.Formatter('%(asctime)s - SECURITY - %(levelname)s - %(message)s')

# Create file handlers
app_handler = logging.FileHandler(settings.APP_LOG_FILE)
app_handler.setLevel(logging.INFO)
app_handler.setFormatter(app_formatter)

audit_handler = logging.FileHandler(settings.AUDIT_LOG_FILE)
audit_handler.setLevel(logging.INFO)
audit_handler.setFormatter(audit_formatter)

security_handler = logging.FileHandler(settings.SECURITY_LOG_FILE)
security_handler.setLevel(logging.INFO)
security_handler.setFormatter(security_formatter)

# Add handlers to logger
logger.addHandler(app_handler)

# Console handler for development
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(app_formatter)
logger.addHandler(console_handler)

def generate_audit_id():
    """Generate a unique ID for audit records"""
    return uuid.uuid4().hex[:12]

def log_audit_event(event_type, details, user=None):
    """
    Log an audit event with standardized format for compliance tracking
    
    Args:
        event_type: Type of event (e.g., "USER_LOGIN", "DOCUMENT_UPLOAD", "QUERY")
        details: Dictionary containing event details
        user: Username if available
    
    Returns:
        Audit ID for reference
    """
    audit_id = generate_audit_id()
    
    # Create audit entry
    audit_entry = {
        "audit_id": audit_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": event_type,
        "user": user if user else "anonymous",
        "details": details
    }
    
    # Write to audit log
    with open(settings.AUDIT_LOG_FILE, "a") as audit_file:
        audit_file.write(json.dumps(audit_entry) + "\n")
    
    # Also log to application log at debug level
    logger.debug(f"Audit event: {event_type} (audit_id: {audit_id})")
    
    return audit_id

def log_security_event(event_type, severity, details, user=None):
    """
    Log a security event for monitoring and alerts
    
    Args:
        event_type: Type of security event (e.g., "AUTHENTICATION_FAILURE", "INVALID_INPUT")
        severity: Severity level (HIGH, MEDIUM, LOW)
        details: Dictionary containing event details
        user: Username if available
    
    Returns:
        Security event ID for reference
    """
    security_id = generate_audit_id()
    
    # Create security entry
    security_entry = {
        "security_id": security_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": event_type,
        "severity": severity,
        "user": user if user else "anonymous",
        "details": details
    }
    
    # Write to security log
    with open(settings.SECURITY_LOG_FILE, "a") as security_file:
        security_file.write(json.dumps(security_entry) + "\n")
    
    # Also log to application log
    log_level = logging.WARNING
    if severity == "HIGH":
        log_level = logging.ERROR
    elif severity == "LOW":
        log_level = logging.INFO
        
    logger.log(log_level, f"Security event: {event_type} severity={severity} (security_id: {security_id})")
    
    return security_id