#!/bin/bash
# Start the Precision Medicine Carriers Pipeline API

# Activate virtual environment
source .venv/bin/activate

# Set environment variables (optional)
export AUTO_OPTIMIZE=true

# Start the API server
echo "🚀 Starting Precision Medicine Carriers Pipeline API..."
echo "📡 API will be available at http://0.0.0.0:8000"
echo "📚 API documentation at http://0.0.0.0:8000/docs"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
