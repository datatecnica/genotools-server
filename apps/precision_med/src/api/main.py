"""
FastAPI application for GP2 Precision Medicine Data Browser.
"""
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..core.config import settings
from .routes import variants

# Create FastAPI application
app = FastAPI(
    title="GP2 Precision Medicine Data Browser",
    description="API for browsing, subsetting, and downloading GP2 genomic and clinical data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(variants.router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint returning basic API information."""
    return {
        "message": "GP2 Precision Medicine Data Browser API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "gp2-precision-med-api",
        "data_root": str(settings.data_root)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 