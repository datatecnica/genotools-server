#!/usr/bin/env python3
"""
Start the Precision Medicine Carriers Pipeline API server.

This script provides a convenient Python interface to start the FastAPI
server with proper configuration and environment setup.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_venv():
    """Check if running in virtual environment."""
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )


def main():
    """Start the API server."""
    # Check virtual environment
    if not check_venv():
        logger.warning("⚠️  Not running in a virtual environment!")
        logger.warning("   Run: source .venv/bin/activate")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Set environment variables
    os.environ.setdefault('AUTO_OPTIMIZE', 'true')

    # Print startup info
    logger.info("🚀 Starting Precision Medicine Carriers Pipeline API...")
    logger.info("📡 API will be available at http://0.0.0.0:8000")
    logger.info("📚 API documentation at http://0.0.0.0:8000/docs")
    logger.info("")

    # Import uvicorn here to ensure proper environment setup
    try:
        import uvicorn
    except ImportError:
        logger.error("❌ uvicorn not found. Install dependencies:")
        logger.error("   pip install -r requirements.txt")
        sys.exit(1)

    # Start the server
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False  # Set to True for development
        )
    except KeyboardInterrupt:
        logger.info("\n🛑 API server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start API server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
