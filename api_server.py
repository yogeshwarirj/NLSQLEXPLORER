"""
NL2SQL API Server - Fixed Version with Session Management
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
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
import traceback
import time

# ==================== CONFIGURATION ====================

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('absl').setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# CORS Configuration
allowed_origins_str = os.getenv('ALLOWED_ORIGINS', '')
if not allowed_origins_str or allowed_origins_str == '*':
    logging.warning("⚠️ CORS set to allow all origins (*)")
    CORS(app)
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',')]
    CORS(app, origins=allowed_origins, supports_credentials=True)
    logging.info(f"✅ CORS configured for: {allowed_origins}")

# Rate Limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Session Management - INCREASED TIMEOUT
sessions: Dict[str, Dict[str, Any]] = {}
sessions_lock = threading.Lock()
SESSION_TIMEOUT = timedelta(hours=8)  # Increased from 2 to 8 hours

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
    """Create secure session ID"""
    hashed_pwd, _ = hash_credentials(password)
    session_data = f"{db_type}:{host}:{database}:{username}:{hashed_pwd}:{secrets.token_hex(8)}"
    return hashlib.sha256(session_data.encode()).hexdigest()

def validate_connection_params(params: dict) -> List[str]:
    """Validate connection parameters"""
    errors = []
    
    host = params.get('host', '')
    if not host or not re.match(r'^[a-zA-Z0-9\.\-]+$', host):
        errors.append("Invalid host format")
    
    port = params.get('port', '')
    try:
        port_int = int(port)
        if port_int < 1 or port_int > 65535:
            errors.append("Port must be between 1-65535")
    except (ValueError, TypeError):
        errors.append("Port must be a valid number")
    
    if not params.get('database'):
        errors.append("Database name is required")
    if not params.get('username'):
        errors.append("Username is required")
    if not params.get('password'):
        errors.append("Password is required")
    
    gemini_api_key = params.get('gemini_api_key', '')
    if not gemini_api_key or not gemini_api_key.startswith('AIza'):
        errors.append("Valid Gemini API key is required")
    
    return errors

# ==================== SQL VALIDATION ====================

def validate_and_sanitize(sql_text: str) -> str:
    """Enhanced SQL validation"""
    sql_text = sql_text.strip()
    
    # Remove comments
    sql_text = re.sub(r'--.*', '', sql_text, flags=re.MULTILINE)
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
    sql_text = re.sub(r'```sql', '', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'```', '', sql_text)
    sql_text = sql_text.strip()
    
    if not sql_text:
        raise ValueError("Empty SQL query")
    
    # Block forbidden keywords
    forbidden_match = FORBIDDEN_KEYWORDS.search(sql_text)
    if forbidden_match:
        raise ValueError(f"Forbidden SQL keyword: {forbidden_match.group()}")
    
    # Single statement only
    statements = sqlparse.split(sql_text)
    if len(statements) > 1:
        raise ValueError("Multiple SQL statements not allowed")
    
    # Block stacked queries
    sql_clean = sql_text.rstrip(';')
    if ';' in sql_clean:
        raise ValueError("Stacked queries not allowed")
    
    # Verify it's SELECT
    parsed = sqlparse.parse(sql_text)
    if not parsed:
        raise ValueError("Invalid SQL syntax")
    
    first_statement = parsed[0]
    statement_type = first_statement.get_type()
    if statement_type and statement_type.upper() != 'SELECT':
        raise ValueError(f"Only SELECT allowed. Got: {statement_type}")
    
    # Add LIMIT
    if not re.search(r'\bLIMIT\b', sql_clean, re.IGNORECASE):
        sql_clean += f" LIMIT {MAX_ROWS}"
    
    return sql_clean + ';'

def create_db_engine(DB_URI: str, db_type: str):
    """Create SQLAlchemy engine with pooling"""
    pool_config = {
        'poolclass': QueuePool,
        'pool_size': 5,
        'max_overflow': 2,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'echo': False
    }
    
    if 'postgresql' in db_type:
        pool_config['connect_args'] = {
            'connect_timeout': 10,
            'options': f'-c statement_timeout={QUERY_TIMEOUT_SECONDS * 1000}'
        }
    elif 'mysql' in db_type:
        pool_config['connect_args'] = {'connect_timeout': 10}
    
    return create_engine(DB_URI, **pool_config)

def execute_with_limits(engine, sql: str) -> Tuple[List, List]:
    """Execute query with limits"""
    with engine.connect() as conn:
        db_url = str(engine.url)
        if 'postgresql' in db_url:
            conn.execute(text(f"SET statement_timeout = {QUERY_TIMEOUT_SECONDS * 1000}"))
        elif 'mysql' in db_url:
            conn.execute(text(f"SET SESSION max_execution_time = {QUERY_TIMEOUT_SECONDS * 1000}"))
        
        result = conn.execute(text(sql))
        columns = list(result.keys())
        
        rows = []
        for i, row in enumerate(result):
            if i >= MAX_ROWS:
                logging.warning(f"Query exceeded {MAX_ROWS} row limit")
                break
            rows.append(list(row))
        
        import sys
        result_size_mb = sys.getsizeof(rows) / (1024 * 1024)
        if result_size_mb > MAX_RESULT_SIZE_MB:
            raise ValueError(f"Result too large: {result_size_mb:.2f}MB")
        
        return rows, columns

def classify_query_type(query: str, llm) -> str:
    """Classify query for chart recommendation"""
    try:
        class_prompt = f"""Classify this query into ONE chart type: bar, line, pie, scatter, or histogram.
Query: "{query}"
Respond with ONLY the chart type."""
        
        response = llm.invoke(class_prompt)
        chart_type = response.content.strip().lower()
        
        type_map = {
            'bar': 'bar', 'ranking': 'bar', 'comparison': 'bar',
            'line': 'line', 'trend': 'line', 'time': 'line',
            'pie': 'pie', 'proportion': 'pie',
            'scatter': 'scatter', 'correlation': 'scatter',
            'histogram': 'histogram'
        }
        return type_map.get(chart_type, 'bar')
    except:
        return 'bar'

# ==================== IMPROVED SESSION MANAGEMENT ====================

def get_session(session_id: str):
    """Get session with expiry check and auto-refresh on activity"""
    with sessions_lock:
        if session_id not in sessions:
            return None
        
        session = sessions[session_id]
        created_at = session.get('created_at', datetime.now())
        last_activity = session.get('last_activity', created_at)
        
        # Check if session has been inactive for too long
        if datetime.now() - last_activity > SESSION_TIMEOUT:
            logging.info(f"Session expired due to inactivity: {session_id[:8]}...")
            try:
                session['engine'].dispose()
            except:
                pass
            del sessions[session_id]
            return None
        
        # Update last activity timestamp to keep session alive
        session['last_activity'] = datetime.now()
        logging.debug(f"Session activity updated: {session_id[:8]}...")
        
        return session

def cleanup_expired_sessions():
    """Remove expired sessions based on last activity"""
    with sessions_lock:
        now = datetime.now()
        expired = []
        
        for sid, data in list(sessions.items()):
            last_activity = data.get('last_activity', data.get('created_at', now))
            if now - last_activity > SESSION_TIMEOUT:
                expired.append(sid)
        
        for sid in expired:
            try:
                sessions[sid]['engine'].dispose()
            except:
                pass
            del sessions[sid]
        
        if expired:
            logging.info(f"Cleaned {len(expired)} expired sessions")

def session_cleanup_worker():
    """Background cleanup worker"""
    while True:
        time.sleep(600)  # Run every 10 minutes
        try:
            cleanup_expired_sessions()
        except Exception as e:
            logging.error(f"Cleanup error: {e}")

cleanup_thread = threading.Thread(target=session_cleanup_worker, daemon=True)
cleanup_thread.start()

# ==================== MIDDLEWARE ====================

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    if hasattr(request, 'start_time'):
        elapsed = time.time() - request.start_time
        response.headers['X-Response-Time'] = f"{elapsed:.3f}s"
    
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Error: {str(e)}, Path: {request.path}")
    return jsonify({"error": str(e)}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded"}), 429

# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(sessions),
        "session_timeout_hours": SESSION_TIMEOUT.total_seconds() / 3600
    })

@app.route('/api/validate-session', methods=['POST'])
@limiter.limit("60 per minute")
def validate_session():
    """Validate if session is still active"""
    try:
        data = request.json or {}
        session_id = data.get('session_id', '').strip()
        
        if not session_id:
            return jsonify({
                "valid": False, 
                "message": "Session ID required"
            }), 400
        
        session = get_session(session_id)
        if not session:
            return jsonify({
                "valid": False, 
                "message": "Invalid or expired session. Please reconnect to the database."
            }), 401
        
        # Session is valid and has been refreshed by get_session()
        return jsonify({
            "valid": True,
            "message": "Session is active",
            "session_info": {
                "db_type": session.get('db_type'),
                "created_at": session.get('created_at').isoformat(),
                "last_activity": session.get('last_activity').isoformat(),
                "expires_in_hours": round((SESSION_TIMEOUT - (datetime.now() - session.get('last_activity'))).total_seconds() / 3600, 2)
            }
        })
    
    except Exception as e:
        logging.error(f"Session validation error: {e}")
        return jsonify({
            "valid": False, 
            "message": "Session validation failed"
        }), 500

@app.route('/api/test-gemini', methods=['POST'])
@limiter.limit("20 per minute")
def test_gemini():
    """Test Gemini API key"""
    try:
        data = request.json
        api_key = data.get('api_key', '').strip()
        
        if not api_key or not api_key.startswith('AIza'):
            return jsonify({"valid": False, "message": "Invalid API key format"}), 400
        
        test_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            google_api_key=api_key,
            timeout=10
        )
        test_llm.invoke("Hello")
        
        return jsonify({"valid": True, "message": "API key is valid"})
    
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            msg = "Rate limit exceeded"
        elif "401" in error_msg or "403" in error_msg:
            msg = "Invalid API key"
        else:
            msg = "Validation failed"
        
        return jsonify({"valid": False, "message": msg}), 400

@app.route('/api/connect-db', methods=['POST'])
@limiter.limit("10 per minute")
def connect_db():
    """Connect to database"""
    try:
        data = request.json or {}
        
        errors = validate_connection_params(data)
        if errors:
            return jsonify({"connected": False, "errors": errors}), 400
        
        db_type = data.get('db_type')
        host = data.get('host')
        port = data.get('port')
        database = data.get('database')
        username = data.get('username')
        password = data.get('password')
        gemini_api_key = data.get('gemini_api_key')
        
        encoded_password = quote_plus(password)
        
        # Get SSL mode (optional)
        ssl_mode = data.get('ssl_mode', 'disable').strip()
        
        if db_type == 'mysql':
            DB_URI = f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database}"
        elif db_type == 'postgresql':
            DB_URI = f"postgresql+psycopg2://{username}:{encoded_password}@{host}:{port}/{database}"
            # Add SSL mode for PostgreSQL if not localhost
            if host not in ['localhost', '127.0.0.1'] or ssl_mode != 'disable':
                DB_URI += f"?sslmode={ssl_mode}"
        elif db_type == 'oracle':
            sid = data.get('sid', '').strip()
            service_name = data.get('service_name', '').strip()
            if sid:
                DB_URI = f"oracle+cx_oracle://{username}:{encoded_password}@{host}:{port}/?sid={sid}"
            elif service_name:
                DB_URI = f"oracle+cx_oracle://{username}:{encoded_password}@{host}:{port}/?service_name={service_name}"
            else:
                return jsonify({"connected": False, "message": "Oracle needs SID or Service Name"}), 400
        else:
            return jsonify({"connected": False, "message": f"Unsupported database type: {db_type}"}), 400
        
        engine = create_db_engine(DB_URI, db_type)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            google_api_key=gemini_api_key
        )
        
        db = SQLDatabase(engine=engine)
        
        # Create SQL query chain
        db_dialect = {"mysql": "MySQL", "postgresql": "PostgreSQL", "oracle": "Oracle"}.get(db_type, "SQL")
        
        template = f"""Given the database schema below, write a SQL query to answer the user's question.
Schema: {{table_info}}

Question: {{input}}

Generate ONLY a valid SELECT query. Rules:
- Use exact column/table names from schema
- Filter for current year if relevant
- Use SUM/GROUP BY/JOIN as needed
- Add LIMIT 100
- No DML operations
- No semicolons

SQL Query:"""

        prompt = PromptTemplate(
            input_variables=["input", "table_info"],
            template=template
        )
        
        # Create the chain
        chain = (
            RunnablePassthrough.assign(table_info=lambda x: db.get_table_info())
            | prompt
            | llm
            | StrOutputParser()
        )
        
        table_names = db.get_usable_table_names()
        
        session_id = create_session_id(db_type, host, database, username, password)
        
        now = datetime.now()
        with sessions_lock:
            sessions[session_id] = {
                "engine": engine,
                "db": db,
                "llm": llm,
                "chain": chain,
                "db_type": db_type,
                "created_at": now,
                "last_activity": now  # Initialize last activity
            }
        
        logging.info(f"✅ Connected: {db_type}@{host}/{database} | Session: {session_id[:8]}...")
        
        return jsonify({
            "connected": True,
            "message": "Connected successfully",
            "session_id": session_id,
            "tables": table_names[:20],
            "table_count": len(table_names),
            "session_timeout_hours": SESSION_TIMEOUT.total_seconds() / 3600
        })
    
    except Exception as e:
        logging.error(f"❌ Connection failed: {e}")
        return jsonify({"connected": False, "message": str(e)}), 400

@app.route('/api/execute-query', methods=['POST'])
@limiter.limit("30 per minute")
def execute_query():
    """Execute natural language query"""
    try:
        data = request.json or {}
        query = data.get('query', '').strip()
        session_id = data.get('session_id', '').strip()
        
        if not query:
            return jsonify({"error": "Query required"}), 400
        if not session_id:
            return jsonify({"error": "Session ID required"}), 400
        
        # Validate session - this also refreshes the session timestamp
        session = get_session(session_id)
        if not session:
            return jsonify({
                "error": "Invalid or expired session",
                "message": "Your session has expired. Please reconnect to the database.",
                "session_expired": True
            }), 401
        
        engine = session['engine']
        llm = session['llm']
        chain = session['chain']
        
        # Generate SQL
        max_retries = 2
        sql = None
        
        for attempt in range(max_retries):
            try:
                sql = chain.invoke({"input": query})
                break
            except Exception as e:
                if ("429" in str(e) or "rate limit" in str(e).lower()) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                elif "429" in str(e):
                    return jsonify({"error": "Gemini API rate limit exceeded. Please try again in a moment."}), 429
                raise
        
        if not sql:
            return jsonify({"error": "Failed to generate SQL query"}), 500
        
        # Validate SQL
        try:
            validated_sql = validate_and_sanitize(sql)
        except ValueError as ve:
            return jsonify({"error": "SQL validation failed", "message": str(ve)}), 400
        
        # Execute
        rows, columns = execute_with_limits(engine, validated_sql)
        
        if not rows:
            return jsonify({
                "success": True,
                "sql": validated_sql,
                "results": {"columns": columns, "rows": [], "row_count": 0},
                "message": "Query executed successfully but returned no results",
                "chart_recommendation": "table"
            })
        
        chart_type = classify_query_type(query, llm)
        
        df = pd.DataFrame(rows, columns=columns)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [col for col in columns if col not in numeric_cols]
        
        logging.info(f"✅ Query executed: {len(rows)} rows | Session: {session_id[:8]}...")
        
        return jsonify({
            "success": True,
            "sql": validated_sql,
            "results": {"columns": columns, "rows": rows, "row_count": len(rows)},
            "chart_recommendation": chart_type,
            "column_info": {"numeric": numeric_cols, "categorical": categorical_cols}
        })
    
    except Exception as e:
        logging.error(f"❌ Query error: {e}")
        return jsonify({"error": "Query execution failed", "details": str(e)}), 500

@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    """Disconnect database"""
    try:
        data = request.json or {}
        session_id = data.get('session_id', '').strip()
        
        if session_id:
            with sessions_lock:
                if session_id in sessions:
                    try:
                        sessions[session_id]['engine'].dispose()
                        logging.info(f"🔌 Disconnected session: {session_id[:8]}...")
                    except Exception as e:
                        logging.error(f"Error disposing engine: {e}")
                    del sessions[session_id]
        
        return jsonify({"success": True, "message": "Disconnected successfully"})
    except Exception as e:
        logging.error(f"Disconnect error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-tables', methods=['POST'])
@limiter.limit("20 per minute")
def get_tables():
    """Get database tables"""
    try:
        data = request.json or {}
        session_id = data.get('session_id', '').strip()
        
        session = get_session(session_id)
        if not session:
            return jsonify({
                "error": "Invalid or expired session",
                "message": "Your session has expired. Please reconnect to the database.",
                "session_expired": True
            }), 401
        
        tables = session['db'].get_usable_table_names()
        return jsonify({"success": True, "tables": tables, "count": len(tables)})
    except Exception as e:
        logging.error(f"Get tables error: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== STARTUP ====================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    
    print("=" * 60)
    print("🚀 NL2SQL API Server Starting...")
    print(f"📍 Port: {port}")
    print(f"🔒 Security: Enhanced")
    print(f"⏱️  Session Timeout: {SESSION_TIMEOUT.total_seconds() / 3600} hours")
    print(f"📊 Max Rows per Query: {MAX_ROWS}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
