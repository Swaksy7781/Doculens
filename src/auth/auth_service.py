"""
Authentication service for the PDF Chat application
Handles user authentication, session management and related security functions
"""

import os
import time
import datetime
import hashlib
import json

from src.logging.log_service import logger, log_audit_event, log_security_event
from src.config import settings

# Simple demo user database - in production, use a secure database
# with properly hashed passwords
DEMO_USERS = {
    "demo": "demo",
    "admin": "admin1234"
}

def check_auth(username, password):
    """
    Basic authentication function for user verification
    
    For a production system, this should be replaced with:
    - A secure database of hashed passwords
    - HTTPS-only connections
    - Multi-factor authentication
    - Account lockout after failed attempts
    
    Args:
        username: The username to check
        password: The password to verify
        
    Returns:
        Boolean indicating if authentication was successful
    """
    # Generate an auth ID for logging
    auth_id = hashlib.md5(f"auth_{time.time()}".encode()).hexdigest()[:8]
    
    logger.info(f"[{auth_id}] Authentication attempt for username: {username}")
    
    # Validate inputs
    if not username or not password or not isinstance(username, str) or not isinstance(password, str):
        logger.warning(f"[{auth_id}] Invalid authentication parameters")
        return False
        
    # Check credentials against demo database
    if username in DEMO_USERS and DEMO_USERS[username] == password:
        logger.info(f"[{auth_id}] Authentication successful for user: {username}")
        
        # Record successful login in audit log
        login_time = datetime.datetime.now().isoformat()
        audit_id = log_audit_event(
            "USER_LOGIN",
            {
                "auth_id": auth_id,
                "username": username,
                "login_time": login_time,
                "ip_address": "127.0.0.1"  # In production, get real IP
            },
            user=username
        )
        
        logger.info(f"[{auth_id}] Login recorded in audit log (audit_id: {audit_id})")
        return True
    else:
        logger.warning(f"[{auth_id}] Authentication failed for username: {username}")
        
        # Record failed login attempt in security log
        security_id = log_security_event(
            "AUTHENTICATION_FAILURE", 
            "MEDIUM",
            {
                "auth_id": auth_id,
                "username": username,
                "attempt_time": datetime.datetime.now().isoformat(),
                "ip_address": "127.0.0.1"  # In production, get real IP
            }
        )
        
        logger.warning(f"[{auth_id}] Failed login recorded in security log (security_id: {security_id})")
        return False

def log_user_logout(username, login_time):
    """
    Log user logout with session information
    
    Args:
        username: The username logging out
        login_time: Timestamp when the user logged in
        
    Returns:
        Audit ID for the logout event
    """
    # Calculate session duration
    logout_time = time.time()
    session_duration = 0
    
    if login_time:
        session_duration = logout_time - login_time
        
    # Format as readable duration
    duration_str = f"{session_duration:.1f} seconds"
    if session_duration > 60:
        duration_str = f"{session_duration/60:.1f} minutes"
    if session_duration > 3600:
        duration_str = f"{session_duration/3600:.1f} hours"
        
    logger.info(f"User {username} logged out after {duration_str}")
    
    # Record logout in audit log
    logout_time_iso = datetime.datetime.now().isoformat()
    audit_id = log_audit_event(
        "USER_LOGOUT",
        {
            "username": username,
            "logout_time": logout_time_iso,
            "session_duration_seconds": session_duration,
            "session_duration_readable": duration_str
        },
        user=username
    )
    
    logger.info(f"Logout recorded in audit log (audit_id: {audit_id})")
    return audit_id