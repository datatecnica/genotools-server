#!/usr/bin/env python3
"""
Start the Precision Medicine Carriers Pipeline API server.

This script provides a convenient Python interface to start the FastAPI
server with proper configuration and environment setup.
"""

import os
import sys
from pathlib import Path

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.core.logging_config import setup_logging, get_progress_logger


def check_venv():
    """Check if running in virtual environment."""
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )


def main():
    """Start the API server."""
    # Setup logging
    setup_logging(job_name="api_server")
    progress = get_progress_logger()

    # Check virtual environment
    if not check_venv():
        progress.warning("Not running in a virtual environment!")
        progress.warning("Run: source .venv/bin/activate")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Set environment variables
    os.environ.setdefault('AUTO_OPTIMIZE', 'true')

    # Print startup info
    print("\nPrecision Medicine API Server")
    print("=" * 35)
    print("API:  http://0.0.0.0:8000")
    print("Docs: http://0.0.0.0:8000/docs")
    print()

    # Import uvicorn here to ensure proper environment setup
    try:
        import uvicorn
    except ImportError:
        progress.error("uvicorn not found. Install dependencies:")
        progress.error("pip install -r requirements.txt")
        sys.exit(1)

    # Start the server
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            log_level="warning",  # Quieter uvicorn logs
            reload=False
        )
    except KeyboardInterrupt:
        print("\nAPI server stopped")
    except Exception as e:
        progress.error(f"Failed to start API server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
