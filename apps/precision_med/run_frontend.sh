#!/bin/bash
# Launch script for Streamlit application

# Activate virtual environment
source .venv/bin/activate

# Parse arguments
PORT=8501
DEBUG_MODE=false

for arg in "$@"; do
    case $arg in
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        [0-9]*)
            PORT=$arg
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Launch with appropriate mode
if [[ "$DEBUG_MODE" == "true" ]]; then
    echo "🔧 Starting app in debug mode on port $PORT..."
    export STREAMLIT_DEBUG=true
    streamlit run app/main.py --server.port $PORT -- --debug
else
    echo "🚀 Starting app in production mode on port $PORT..."
    export STREAMLIT_DEBUG=false
    streamlit run app/main.py --server.port $PORT
fi