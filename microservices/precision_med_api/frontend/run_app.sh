#!/bin/bash
# Launch script for Streamlit application

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment (located in parent directory)
source "$SCRIPT_DIR/../.venv/bin/activate"

# Change to frontend directory for relative imports
cd "$SCRIPT_DIR"

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
    echo "🔧 Starting app in debug mode on port $PORT (auto-reload enabled)..."
    export STREAMLIT_DEBUG=true
    streamlit run app/main.py --server.port $PORT --server.runOnSave true --server.fileWatcherType auto -- --debug
else
    echo "🚀 Starting app in production mode on port $PORT..."
    export STREAMLIT_DEBUG=false
    streamlit run app/main.py --server.port $PORT
fi
