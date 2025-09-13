#!/bin/bash
# Launch script for Streamlit carriers pipeline viewer

echo "🧬 Starting Carriers Pipeline Results Viewer..."
echo "📁 Results path: ~/gcs_mounts/genotools_server/precision_med/results"

# Check for debug mode
if [[ "$1" == "--debug" ]]; then
    echo "🔧 Debug mode enabled - Job selection will be available"
    DEBUG_ARGS="-- --debug"
else
    echo "📊 Production mode - Using release-level results only"
    DEBUG_ARGS=""
fi

echo ""

# Launch streamlit
echo "🚀 Launching Streamlit app..."
echo "Access at: http://localhost:8501"
echo ""

streamlit run streamlit_viewer.py --server.address=0.0.0.0 --server.port=8501 $DEBUG_ARGS