from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, text
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
import json
from typing import Dict, Any
import traceback

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('absl').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

# Store active sessions (use Redis in production)
sessions: Dict[str, Dict[str, Any]] = {}

# SQL validation regex
FORBIDDEN = re.compile(
    r'\b(insert|update|delete|drop|alter|create|truncate|grant|revoke)\b', re.I
)

def validate_and_sanitize(sql_text: str) -> str:
    """Validate and sanitize SQL for safety"""
    sql_text = sql_text.strip()
    
    if FORBIDDEN.search(sql_text):
        raise ValueError("Forbidden SQL keyword detected! Only SELECT allowed.")
    
    statements = sqlparse.split(sql_text)
    if len(statements) > 1:
        raise ValueError("Multiple statements detected! Only single SELECT allowed.")
    
    parsed = sqlparse.parse(sql_text)[0]
    if parsed.get_type().upper() != 'SELECT':
        raise ValueError("Only SELECT statements are allowed!")
    
    sql_text = sql_text.rstrip(';')
    if not re.search(r'\bLIMIT\b', sql_text, re.I):
        sql_text += " LIMIT 100"
    
    return sql_text + ';'

def classify_query_type(query: str, llm) -> str:
    """Classify query for chart recommendation"""
    class_prompt = f"""
    Classify this database query into a chart type: bar (rankings/comparisons), line (trends over time), 
    pie (proportions/breakdowns), scatter (correlations), histogram (distributions), or table (other).
    Query: "{query}"
    Respond with ONLY the chart type (e.g., 'bar')."""
    
    try:
        response = llm.invoke(class_prompt)
        chart_type = response.content.strip().lower()
        
        type_map = {
            'bar': 'bar', 'ranking': 'bar', 'comparison': 'bar', 'top': 'bar',
            'line': 'line', 'trend': 'line', 'over time': 'line', 'monthly': 'line',
            'pie': 'pie', 'proportion': 'pie', 'breakdown': 'pie', 'share': 'pie',
            'scatter': 'scatter', 'correlation': 'scatter',
            'histogram': 'histogram', 'distribution': 'histogram',
        }
        return type_map.get(chart_type, 'bar')
    except:
        return 'bar'

# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"})

@app.route('/api/test-gemini', methods=['POST'])
def test_gemini():
    """Test Gemini API key validity"""
    try:
        data = request.json
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify({"valid": False, "message": "API key is required"}), 400
        
        # Test the API key
        test_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            google_api_key=api_key
        )
        response = test_llm.invoke("Hello")
        
        return jsonify({
            "valid": True,
            "message": "API key is valid and working"
        })
    
    except Exception as e:
        return jsonify({
            "valid": False,
            "message": f"Invalid API key: {str(e)}"
        }), 400

@app.route('/api/connect-db', methods=['POST'])
def connect_db():
    """Test and establish database connection"""
    try:
        data = request.json
        db_type = data.get('db_type')
        host = data.get('host', 'localhost')
        port = data.get('port')
        database = data.get('database')
        username = data.get('username')
        password = data.get('password')
        gemini_api_key = data.get('gemini_api_key')
        
        # Oracle specific
        sid = data.get('sid', '')
        service_name = data.get('service_name', '')
        
        # Validate inputs
        if not all([db_type, host, port, database, username, password, gemini_api_key]):
            return jsonify({
                "connected": False,
                "message": "Missing required connection parameters"
            }), 400
        
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
        
        # Test connection
        engine = create_engine(DB_URI)
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
        
        # Create prompt template based on db_type
        if db_type == 'mysql':
            db_dialect = "MySQL"
        elif db_type == 'postgresql':
            db_dialect = "PostgreSQL"
        else:
            db_dialect = "Oracle"
        
        PROMPT = PromptTemplate(
            input_variables=["input", "table_info"],
            template=f"""You are a {db_dialect} expert. Given the schema: {{table_info}}

Question: {{input}}

Generate ONLY a valid SELECT query. Use exact column/table names from schema.
- Filter for current year if relevant (e.g., date >= '2025-01-01').
- Aggregate with SUM/GROUP BY/JOIN as needed.
- Always add LIMIT 100.
- No DML (INSERT/UPDATE/DELETE/DROP/TRUNCATE). No semicolons at end.

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
        
        # Create session
        session_id = f"{host}_{database}_{hash(password)}"
        sessions[session_id] = {
            "engine": engine,
            "db": db,
            "llm": llm,
            "db_chain": db_chain,
            "db_type": db_type,
            "gemini_api_key": gemini_api_key,
            "connection_info": {
                "host": host,
                "port": port,
                "database": database,
                "username": username
            }
        }
        
        return jsonify({
            "connected": True,
            "message": "Connected successfully",
            "session_id": session_id,
            "tables": table_names[:20],  # Return first 20 tables
            "table_count": len(table_names)
        })
    
    except Exception as e:
        logging.error(f"Connection error: {traceback.format_exc()}")
        return jsonify({
            "connected": False,
            "message": f"Connection failed: {str(e)}"
        }), 400

@app.route('/api/execute-query', methods=['POST'])
def execute_query():
    """Execute natural language query"""
    try:
        data = request.json
        query = data.get('query')
        session_id = data.get('session_id')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        if not session_id or session_id not in sessions:
            return jsonify({
                "error": "Invalid or expired session. Please reconnect to database."
            }), 401
        
        session = sessions[session_id]
        engine = session['engine']
        db = session['db']
        llm = session['llm']
        db_chain = session['db_chain']
        
        # Generate SQL
        sql_prompt = db_chain.llm_chain.prompt.format(
            input=query,
            table_info=db.get_table_info()
        )
        sql_response = llm.invoke(sql_prompt)
        sql = sql_response.content.strip()
        
        # Validate and sanitize
        validated_sql = validate_and_sanitize(sql)
        
        # Execute query
        with engine.connect() as conn:
            result = conn.execute(text(validated_sql))
            raw_results = result.fetchall()
            columns = list(result.keys())
        
        # Convert to JSON-serializable format
        rows = [list(row) for row in raw_results]
        
        # Classify query for chart recommendation
        chart_type = classify_query_type(query, llm)
        
        # Auto-detect column types for better visualization
        df = pd.DataFrame(rows, columns=columns)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [col for col in columns if col not in numeric_cols]
        
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
        logging.error(f"Query execution error: {traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "message": "Query execution failed. Please check your query and try again."
        }), 500

@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    """Disconnect from database"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id and session_id in sessions:
            # Close engine connection
            sessions[session_id]['engine'].dispose()
            del sessions[session_id]
            
        return jsonify({
            "success": True,
            "message": "Disconnected successfully"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/get-tables', methods=['POST'])
def get_tables():
    """Get list of tables in connected database"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id or session_id not in sessions:
            return jsonify({"error": "Invalid session"}), 401
        
        session = sessions[session_id]
        db = session['db']
        tables = db.get_usable_table_names()
        
        return jsonify({
            "tables": tables
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == '__main__':
    print("🚀 NL2SQL API Server Starting...")
    print("📍 Backend URL: http://localhost:5000")
    print("📚 API Docs: http://localhost:5000/api/health")
    app.run(debug=True, host='0.0.0.0', port=5000)
