"""
NL2SQL API Server - Production Secured Version
Features: SQL injection protection, rate limiting, session management, logging sanitization
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from urllib.parse import quote_plus
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.prompts import PromptTemplate
import os
import re
import logging
import sqlparse
import pandas as pd
import hashlib
import secrets
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List
from functools import wraps
import traceback
import time

# ==================== CONFIGURATION ====================

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('absl').setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request

# CORS Configuration - PRODUCTION READY
allowed_origins_str = os.getenv('ALLOWED_ORIGINS', '')
if not allowed_origins_str or allowed_origins_str == '*':
    logging.warning("⚠️ CORS set to allow all origins (*). Insecure for production!")
    CORS(app)
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',')]
    CORS(app, 
         origins=allowed_origins,
         supports_credentials=True,
         allow_headers=["Content-Type", "Authorization", "X-API-Key"],
         methods=["GET", "POST", "OPTIONS"])
    logging.info(f"✅ CORS configured for: {allowed_origins}")

# Rate Limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"  # Use Redis in production: redis://localhost:6379
)

# Session Management
sessions: Dict[str, Dict[str, Any]] = {}
sessions_lock = threading.Lock()
SESSION_TIMEOUT = timedelta(hours=2)

# Query Limits
MAX_ROWS = 10000
MAX_RESULT_SIZE_MB = 50
QUERY_TIMEOUT_SECONDS = 30

# SQL Validation
FORBIDDEN_KEYWORDS = re.compile(
    r'\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|'
    r'exec|execute|script|declare|cast|convert|union|into|outfile|'
    r'dumpfile|load_file|benchmark|sleep|waitfor)\b',
    re.IGNORECASE
)

# ==================== SECURITY UTILITIES ====================

def sanitize_for_logging(data):
    """Remove sensitive data from logs"""
    if isinstance(data, dict):
        sanitized = {}
        sensitive_keys = ['password', 'api_key', 'gemini_api_key', 'token', 'secret']
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '***REDACTED***'
            else:
                sanitized[key] = sanitize_for_logging(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_for_logging(item) for item in data]
    return data

def hash_credentials(password: str, salt: str = None) -> tuple:
    """Hash password for session ID generation"""
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return hashed.hex(), salt

def create_session_id(db_type: str, host: str, database: str, username: str, password: str) -> str:
    """Create secure session ID without storing password"""
    hashed_pwd, _ = hash_credentials(password)
    session_data = f"{db_type}:{host}:{database}:{username}:{hashed_pwd}:{secrets.token_hex(8)}"
    return hashlib.sha256(session_data.encode()).hexdigest()

def validate_connection_params(params: dict) -> List[str]:
    """Validate connection parameters"""
    errors = []
    
    host = params.get('host', '')
    if not host or not re.match(r'^[a-zA-Z0-9\.\-]+, host):
        errors.append("Invalid host format (alphanumeric, dots, hyphens only)")
    
    port = params.get('port', '')
    try:
        port_int = int(port)
        if port_int < 1 or port_int > 65535:
            errors.append("Port must be between 1-65535")
    except (ValueError, TypeError):
        errors.append("Port must be a valid number")
    
    database = params.get('database', '')
    if not database:
        errors.append("Database name is required")
    
    username = params.get('username', '')
    if not username:
        errors.append("Username is required")
    
    password = params.get('password', '')
    if not password:
        errors.append("Password is required")
    
    gemini_api_key = params.get('gemini_api_key', '')
    if not gemini_api_key or not gemini_api_key.startswith('AIza'):
        errors.append("Valid Gemini API key is required")
    
    return errors

# ==================== SQL VALIDATION & EXECUTION ====================

def validate_and_sanitize(sql_text: str) -> str:
    """Enhanced SQL validation with multiple security layers"""
    sql_text = sql_text.strip()
    
    # Layer 1: Remove SQL comments (prevent comment-based bypasses)
    sql_text = re.sub(r'--.*, '', sql_text, flags=re.MULTILINE)
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
    sql_text = sql_text.strip()
    
    if not sql_text:
        raise ValueError("Empty SQL query after removing comments")
    
    # Layer 2: Block forbidden keywords
    forbidden_match = FORBIDDEN_KEYWORDS.search(sql_text)
    if forbidden_match:
        raise ValueError(f"Forbidden SQL keyword detected: {forbidden_match.group()}")
    
    # Layer 3: Ensure only one statement
    statements = sqlparse.split(sql_text)
    if len(statements) > 1:
        raise ValueError("Multiple SQL statements detected. Only single SELECT allowed.")
    
    # Layer 4: Block stacked queries (semicolons)
    sql_clean = sql_text.rstrip(';')
    if ';' in sql_clean:
        raise ValueError("Stacked queries not allowed (multiple semicolons detected)")
    
    # Layer 5: Parse and verify it's SELECT
    parsed = sqlparse.parse(sql_text)
    if not parsed or len(parsed) == 0:
        raise ValueError("Invalid SQL syntax")
    
    first_statement = parsed[0]
    statement_type = first_statement.get_type()
    if statement_type and statement_type.upper() != 'SELECT':
        raise ValueError(f"Only SELECT statements allowed. Got: {statement_type}")
    
    # Layer 6: Add LIMIT if missing
    if not re.search(r'\bLIMIT\b', sql_clean, re.IGNORECASE):
        sql_clean += f" LIMIT {MAX_ROWS}"
    
    return sql_clean + ';'

def create_db_engine(DB_URI: str, db_type: str):
    """Create SQLAlchemy engine with proper pooling and timeouts"""
    
    pool_config = {
        'poolclass': QueuePool,
        'pool_size': 5,
        'max_overflow': 2,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'echo': False
    }
    
    # Add connection timeout
    if 'postgresql' in db_type:
        pool_config['connect_args'] = {
            'connect_timeout': 10,
            'options': f'-c statement_timeout={QUERY_TIMEOUT_SECONDS * 1000}'
        }
    elif 'mysql' in db_type:
        pool_config['connect_args'] = {
            'connect_timeout': 10
        }
    
    return create_engine(DB_URI, **pool_config)

def execute_with_limits(engine, sql: str) -> Tuple[List, List]:
    """Execute query with size and timeout limits"""
    with engine.connect() as conn:
        # Set timeout (database-specific)
        db_url = str(engine.url)
        if 'postgresql' in db_url:
            conn.execute(text(f"SET statement_timeout = {QUERY_TIMEOUT_SECONDS * 1000}"))
        elif 'mysql' in db_url:
            conn.execute(text(f"SET SESSION max_execution_time = {QUERY_TIMEOUT_SECONDS * 1000}"))
        
        # Execute query
        result = conn.execute(text(sql))
        columns = list(result.keys())
        
        # Fetch with row limit
        rows = []
        for i, row in enumerate(result):
            if i >= MAX_ROWS:
                logging.warning(f"Query exceeded {MAX_ROWS} row limit, truncating results")
                break
            rows.append(list(row))
        
        # Check result size
        import sys
        result_size_bytes = sys.getsizeof(rows)
        result_size_mb = result_size_bytes / (1024 * 1024)
        
        if result_size_mb > MAX_RESULT_SIZE_MB:
            raise ValueError(f"Result too large: {result_size_mb:.2f}MB. Max: {MAX_RESULT_SIZE_MB}MB")
        
        return rows, columns

def classify_query_type(query: str, llm) -> str:
    """Classify query for chart recommendation"""
    try:
        class_prompt = f"""Classify this database query into ONE chart type: bar, line, pie, scatter, or histogram.
Query: "{query}"
Respond with ONLY the chart type (one word)."""
        
        response = llm.invoke(class_prompt)
        chart_type = response.content.strip().lower()
        
        type_map = {
            'bar': 'bar', 'ranking': 'bar', 'comparison': 'bar', 'top': 'bar',
            'line': 'line', 'trend': 'line', 'time': 'line',
            'pie': 'pie', 'proportion': 'pie', 'distribution': 'pie',
            'scatter': 'scatter', 'correlation': 'scatter',
            'histogram': 'histogram'
        }
        return type_map.get(chart_type, 'bar')
    except Exception as e:
        logging.error(f"Chart classification error: {e}")
        return 'bar'

# ==================== SESSION MANAGEMENT ====================

def get_session(session_id: str):
    """Thread-safe session retrieval with expiry check"""
    with sessions_lock:
        if session_id not in sessions:
            return None
        
        session = sessions[session_id]
        created_at = session.get('created_at', datetime.now())
        
        # Check expiry
        if datetime.now() - created_at > SESSION_TIMEOUT:
            logging.info(f"Session {session_id[:8]}... expired")
            try:
                session['engine'].dispose()
            except:
                pass
            del sessions[session_id]
            return None
        
        return session

def cleanup_expired_sessions():
    """Remove expired sessions"""
    with sessions_lock:
        now = datetime.now()
        expired = []
        
        for sid, data in list(sessions.items()):
            created_at = data.get('created_at', now)
            if now - created_at > SESSION_TIMEOUT:
                expired.append(sid)
        
        for sid in expired:
            try:
                sessions[sid]['engine'].dispose()
            except:
                pass
            del sessions[sid]
        
        if expired:
            logging.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)

def session_cleanup_worker():
    """Background worker to cleanup expired sessions"""
    while True:
        time.sleep(600)  # Every 10 minutes
        try:
            cleanup_expired_sessions()
        except Exception as e:
            logging.error(f"Session cleanup error: {e}")

# Start cleanup worker
cleanup_thread = threading.Thread(target=session_cleanup_worker, daemon=True)
cleanup_thread.start()

# ==================== MIDDLEWARE ====================

@app.before_request
def before_request():
    """Track request start time"""
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """Add security headers and timing"""
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # Response timing
    if hasattr(request, 'start_time'):
        elapsed = time.time() - request.start_time
        response.headers['X-Response-Time'] = f"{elapsed:.3f}s"
        
        if elapsed > 5:
            logging.warning(f"Slow request: {request.path} took {elapsed:.2f}s")
    
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler with sanitized logging"""
    request_data = {}
    try:
        if request.is_json:
            request_data = sanitize_for_logging(request.json)
    except:
        pass
    
    logging.error(f"Error: {str(e)}, Path: {request.path}, Data: {request_data}")
    
    # Return user-friendly error
    return jsonify({
        "error": "An error occurred processing your request",
        "message": str(e)
    }), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    """Rate limit error handler"""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again in a few moments."
    }), 429

# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check with dependency status"""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(sessions),
        "checks": {}
    }
    
    # Check Gemini API
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            test_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0,
                google_api_key=api_key
            )
            test_llm.invoke("test")
            health["checks"]["gemini_api"] = "ok"
        else:
            health["checks"]["gemini_api"] = "not_configured"
    except Exception as e:
        health["checks"]["gemini_api"] = f"error"
        health["status"] = "degraded"
        logging.error(f"Gemini API health check failed: {e}")
    
    status_code = 200 if health["status"] == "healthy" else 503
    return jsonify(health), status_code

@app.route('/api/test-gemini', methods=['POST'])
@limiter.limit("20 per minute")
def test_gemini():
    """Test Gemini API key validity"""
    try:
        data = request.json
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({"valid": False, "message": "API key is required"}), 400
        
        if not api_key.startswith('AIza'):
            return jsonify({"valid": False, "message": "Invalid API key format"}), 400
        
        # Test the API key with timeout
        test_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            google_api_key=api_key,
            timeout=10
        )
        response = test_llm.invoke("Hello")
        
        logging.info("Gemini API key validated successfully")
        return jsonify({
            "valid": True,
            "message": "API key is valid and working"
        })
    
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate" in error_msg.lower():
            msg = "API key valid but rate limit exceeded. Try again in a moment."
        elif "401" in error_msg or "403" in error_msg or "invalid" in error_msg.lower():
            msg = "Invalid API key. Please check and try again."
        else:
            msg = "API key validation failed. Please verify and try again."
        
        logging.warning(f"Gemini API test failed: {error_msg}")
        return jsonify({"valid": False, "message": msg}), 400

@app.route('/api/connect-db', methods=['POST'])
@limiter.limit("10 per minute")
def connect_db():
    """Test and establish database connection"""
    try:
        data = request.json or {}
        
        # Validate input parameters
        errors = validate_connection_params(data)
        if errors:
            return jsonify({
                "connected": False,
                "message": "Validation failed",
                "errors": errors
            }), 400
        
        db_type = data.get('db_type')
        host = data.get('host')
        port = data.get('port')
        database = data.get('database')
        username = data.get('username')
        password = data.get('password')
        gemini_api_key = data.get('gemini_api_key')
        
        # Oracle specific
        sid = data.get('sid', '').strip()
        service_name = data.get('service_name', '').strip()
        
        # Build connection string
        encoded_password = quote_plus(password)
        
        if db_type == 'mysql':
            DB_URI = f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database}"
        elif db_type == 'postgresql':
            DB_URI = f"postgresql+psycopg2://{username}:{encoded_password}@{host}:{port}/{database}"
        elif db_type == 'oracle':
            if sid:
                DB_URI = f"oracle+cx_oracle://{username}:{encoded_password}@{host}:{port}/?sid={sid}"
            elif service_name:
                DB_URI = f"oracle+cx_oracle://{username}:{encoded_password}@{host}:{port}/?service_name={service_name}"
            else:
                return jsonify({
                    "connected": False,
                    "message": "Oracle requires either SID or Service Name"
                }), 400
        else:
            return jsonify({
                "connected": False,
                "message": f"Unsupported database type: {db_type}"
            }), 400
        
        # Create engine with pooling
        engine = create_db_engine(DB_URI, db_type)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Initialize LangChain components
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            google_api_key=gemini_api_key
        )
        
        db = SQLDatabase(engine=engine)
        
        # Create database-specific prompt
        db_dialect = {"mysql": "MySQL", "postgresql": "PostgreSQL", "oracle": "Oracle"}.get(db_type, "SQL")
        
        PROMPT = PromptTemplate(
            input_variables=["input", "table_info"],
            template=f"""You are a {db_dialect} expert. Given the schema: {{table_info}}

Question: {{input}}

Generate ONLY a valid SELECT query. Use exact column/table names from schema.
- Filter for current year if relevant (e.g., date >= '2025-01-01').
- Aggregate with SUM/GROUP BY/JOIN as needed.
- Always add LIMIT 100.
- No DML (INSERT/UPDATE/DELETE/DROP/TRUNCATE).
- No semicolons at end.

SQL Query: give only query dont start with ```sql and end with ```"""
        )
        
        db_chain = SQLDatabaseChain.from_llm(
            llm=llm,
            db=db,
            prompt=PROMPT,
            verbose=False
        )
        
        # Get table info
        table_names = db.get_usable_table_names()
        
        # Create secure session (no password stored!)
        session_id = create_session_id(db_type, host, database, username, password)
        
        with sessions_lock:
            sessions[session_id] = {
                "engine": engine,
                "db": db,
                "llm": llm,
                "db_chain": db_chain,
                "db_type": db_type,
                "created_at": datetime.now(),
                "connection_info": {
                    "host": host,
                    "port": port,
                    "database": database,
                    "username": username
                    # ❌ NO PASSWORD STORED
                }
            }
        
        logging.info(f"Database connection established: {db_type}@{host}/{database}")
        
        return jsonify({
            "connected": True,
            "message": "Connected successfully",
            "session_id": session_id,
            "tables": table_names[:20],
            "table_count": len(table_names)
        })
    
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Connection failed: {error_msg}")
        
        # User-friendly error messages
        if "timeout" in error_msg.lower():
            msg = "Connection timeout. Please check host and port."
        elif "access denied" in error_msg.lower() or "authentication" in error_msg.lower():
            msg = "Authentication failed. Please check username and password."
        elif "unknown database" in error_msg.lower():
            msg = "Database not found. Please check database name."
        else:
            msg = f"Connection failed: {error_msg}"
        
        return jsonify({
            "connected": False,
            "message": msg
        }), 400

@app.route('/api/execute-query', methods=['POST'])
@limiter.limit("30 per minute")
def execute_query():
    """Execute natural language query"""
    try:
        data = request.json or {}
        query = data.get('query', '').strip()
        session_id = data.get('session_id', '').strip()
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        # Get session with expiry check
        session = get_session(session_id)
        if not session:
            return jsonify({
                "error": "Invalid or expired session. Please reconnect to database."
            }), 401
        
        engine = session['engine']
        db = session['db']
        llm = session['llm']
        db_chain = session['db_chain']
        
        # Generate SQL with retry for rate limits
        max_retries = 2
        sql = None
        
        for attempt in range(max_retries):
            try:
                sql_prompt = db_chain.llm_chain.prompt.format(
                    input=query,
                    table_info=db.get_table_info()
                )
                sql_response = llm.invoke(sql_prompt)
                sql = sql_response.content.strip()
                break
            except Exception as e:
                if ("429" in str(e) or "rate limit" in str(e).lower()) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                elif "429" in str(e) or "rate limit" in str(e).lower():
                    return jsonify({
                        "error": "Gemini API rate limit exceeded. Please wait a moment and try again."
                    }), 429
                raise
        
        if not sql:
            return jsonify({"error": "Failed to generate SQL"}), 500
        
        # Validate and sanitize SQL
        try:
            validated_sql = validate_and_sanitize(sql)
        except ValueError as ve:
            logging.warning(f"SQL validation failed: {ve}")
            return jsonify({
                "error": "Generated SQL failed security validation",
                "message": str(ve)
            }), 400
        
        # Execute query with limits
        rows, columns = execute_with_limits(engine, validated_sql)
        
        # Handle empty results
        if not rows or len(rows) == 0:
            return jsonify({
                "success": True,
                "sql": validated_sql,
                "results": {
                    "columns": columns if columns else [],
                    "rows": [],
                    "row_count": 0
                },
                "message": "Query executed successfully but returned no results",
                "chart_recommendation": "table",
                "column_info": {
                    "numeric": [],
                    "categorical": []
                }
            })
        
        # Classify query for chart
        chart_type = classify_query_type(query, llm)
        
        # Detect column types
        df = pd.DataFrame(rows, columns=columns)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [col for col in columns if col not in numeric_cols]
        
        logging.info(f"Query executed successfully: {len(rows)} rows returned")
        
        return jsonify({
            "success": True,
            "sql": validated_sql,
            "results": {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows)
            },
            "chart_recommendation": chart_type,
            "column_info": {
                "numeric": numeric_cols,
                "categorical": categorical_cols
            }
        })
    
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Query execution error: {error_msg}")
        
        # User-friendly error messages
        if "timeout" in error_msg.lower():
            msg = "Query timed out. Please simplify your query."
        elif "syntax" in error_msg.lower():
            msg = "SQL syntax error. Please rephrase your question."
        else:
            msg = "Query execution failed. Please try again or rephrase your question."
        
        return jsonify({
            "error": msg,
            "details": error_msg
        }), 500

@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    """Disconnect from database and cleanup session"""
    try:
        data = request.json or {}
        session_id = data.get('session_id', '').strip()
        
        if session_id:
            with sessions_lock:
                if session_id in sessions:
                    try:
                        sessions[session_id]['engine'].dispose()
                    except:
                        pass
                    del sessions[session_id]
                    logging.info(f"Session {session_id[:8]}... disconnected")
        
        return jsonify({
            "success": True,
            "message": "Disconnected successfully"
        })
    
    except Exception as e:
        logging.error(f"Disconnect error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-tables', methods=['POST'])
@limiter.limit("20 per minute")
def get_tables():
    """Get list of tables in connected database"""
    try:
        data = request.json or {}
        session_id = data.get('session_id', '').strip()
        
        session = get_session(session_id)
        if not session:
            return jsonify({"error": "Invalid or expired session"}), 401
        
        db = session['db']
        tables = db.get_usable_table_names()
        
        return jsonify({
            "success": True,
            "tables": tables,
            "count": len(tables)
        })
    
    except Exception as e:
        logging.error(f"Get tables error: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== SERVER STARTUP ====================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    
    print("=" * 60)
    print("🚀 NL2SQL API Server Starting...")
    print(f"📍 Backend URL: http://{'localhost' if debug else '0.0.0.0'}:{port}")
    print(f"🔒 Security: Enhanced (SQL injection protection, rate limiting)")
    print(f"⏱️  Session timeout: {SESSION_TIMEOUT.total_seconds()/3600:.1f} hours")
    print(f"📊 Max result size: {MAX_RESULT_SIZE_MB}MB, {MAX_ROWS} rows")
    print(f"🌐 CORS: {os.getenv('ALLOWED_ORIGINS', 'All origins (*)')}")
    print("=" * 60)
    
    # Production: Use gunicorn
    # Development: Use Flask dev server
    app.run(host='0.0.0.0', port=port, debug=debug)
