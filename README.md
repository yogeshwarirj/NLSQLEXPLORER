# NL2SQL API Server

Natural Language to SQL query API powered by LangChain and Google Gemini.

## Environment Variables

- `GOOGLE_API_KEY` - Google Gemini API key
- `FLASK_ENV` - Environment (development/production)
- `ALLOWED_ORIGINS` - Comma-separated list of allowed CORS origins
- `PORT` - Server port (auto-set by Render)

## Deployment

Deployed on Render.com

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/test-gemini` - Test Gemini API key
- `POST /api/connect-db` - Connect to database
- `POST /api/execute-query` - Execute natural language query
