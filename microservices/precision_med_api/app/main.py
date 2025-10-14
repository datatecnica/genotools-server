"""
FastAPI application for Precision Medicine Carriers Pipeline API.

Provides RESTful API endpoints for executing the carriers pipeline
with the same functionality as the CLI script.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .core.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting Precision Medicine Carriers Pipeline API")

    # Validate settings
    try:
        settings = Settings.create_optimized()
        logger.info(f"✅ Settings validated: Release {settings.release}")
        logger.info(f"📁 Results path: {settings.results_path}")
        logger.info(f"⚙️  Performance: {settings.max_workers} workers, "
                   f"{settings.chunk_size} chunk_size")
    except Exception as e:
        logger.error(f"❌ Settings validation failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down Precision Medicine Carriers Pipeline API")


# Create FastAPI app
app = FastAPI(
    title="Precision Medicine Carriers Pipeline API",
    description=(
        "RESTful API for executing genomic carrier screening pipeline "
        "across multiple data types (NBA, WGS, IMPUTED) with harmonization, "
        "probe selection, and locus report generation."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1/carriers", tags=["carriers"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Precision Medicine Carriers Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/carriers/health"
    }


if __name__ == "__main__":
    import uvicorn

    # Run with uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )
