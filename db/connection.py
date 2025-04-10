import os
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database configuration from environment variables
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5432")
DB_NAME = os.getenv("PGDATABASE", "pdf_chat")
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD", "")
DB_URL = os.getenv("DATABASE_URL", "")

# Create a connection pool
try:
    # Try to use the DATABASE_URL if available
    if DB_URL:
        connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DB_URL
        )
    else:
        # Fallback to individual parameters
        connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    logger.info("Database connection pool created successfully")
except Exception as e:
    logger.error(f"Error creating connection pool: {e}")
    connection_pool = None


def init_db():
    """Initialize the database by creating the schema if it doesn't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if pgvector extension exists, install if not
        cursor.execute("SELECT COUNT(*) FROM pg_available_extensions WHERE name = 'vector'")
        if cursor.fetchone()[0] == 1:  # Extension is available
            cursor.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'")
            if cursor.fetchone()[0] == 0:  # Extension is not installed
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                logger.info("pgvector extension installed successfully")
        
        # Read and execute schema SQL
        with open('db/schema.sql', 'r') as schema_file:
            schema_sql = schema_file.read()
            cursor.execute(schema_sql)
        
        conn.commit()
        logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database schema: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            release_db_connection(conn)


def get_db_connection():
    """Get a connection from the pool."""
    if connection_pool:
        try:
            conn = connection_pool.getconn()
            conn.autocommit = False
            return conn
        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}")
            return None
    else:
        logger.error("Connection pool not available")
        return None


def release_db_connection(conn):
    """Release a connection back to the pool."""
    if connection_pool and conn:
        try:
            connection_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error releasing connection back to pool: {e}")


def execute_query(query, params=None, fetch_one=False, fetch_all=False, dict_cursor=True):
    """
    Execute a query and handle connection/cursor management.
    
    Args:
        query (str): SQL query to execute
        params (tuple, optional): Parameters for the query
        fetch_one (bool): Whether to fetch a single row
        fetch_all (bool): Whether to fetch all rows
        dict_cursor (bool): Whether to use a dictionary cursor
    
    Returns:
        The query result or None in case of error
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return None
        
        if dict_cursor:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = conn.cursor()
        
        cursor.execute(query, params)
        
        if fetch_one:
            result = cursor.fetchone()
        elif fetch_all:
            result = cursor.fetchall()
        else:
            result = cursor.rowcount
        
        conn.commit()
        return result
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            release_db_connection(conn)


def close_connection_pool():
    """Close the connection pool when the application terminates."""
    if connection_pool:
        try:
            connection_pool.closeall()
            logger.info("Connection pool closed")
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")
