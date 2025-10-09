"""
API request/response models for carriers pipeline endpoints.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineRequest(BaseModel):
    """Request model for carriers pipeline execution."""

    job_name: str = Field(
        default="carriers_analysis",
        description="Job name for output files"
    )

    ancestries: Optional[List[str]] = Field(
        default=None,
        description="Ancestries to process (default: all from config)"
    )

    data_types: List[str] = Field(
        default=["NBA", "WGS", "IMPUTED"],
        description="Data types to process"
    )

    parallel: bool = Field(
        default=True,
        description="Enable parallel processing"
    )

    max_workers: Optional[int] = Field(
        default=None,
        description="Maximum workers (default: auto-detect)"
    )

    optimize: bool = Field(
        default=True,
        description="Use performance optimizations"
    )

    skip_extraction: bool = Field(
        default=False,
        description="Skip extraction phase if results already exist"
    )

    skip_probe_selection: bool = Field(
        default=False,
        description="Skip probe selection phase"
    )

    skip_locus_reports: bool = Field(
        default=False,
        description="Skip locus report generation"
    )

    output_dir: Optional[str] = Field(
        default=None,
        description="Custom output directory (default: config-based results path)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_name": "my_analysis",
                "ancestries": ["AAC", "AFR", "EUR"],
                "data_types": ["NBA", "WGS"],
                "parallel": True,
                "optimize": True,
                "skip_extraction": False,
                "skip_probe_selection": False,
                "skip_locus_reports": False
            }
        }


class PipelineResponse(BaseModel):
    """Response model for pipeline execution."""

    success: bool = Field(description="Whether pipeline completed successfully")
    job_id: str = Field(description="Unique job identifier")
    status: JobStatus = Field(description="Current job status")
    message: str = Field(description="Status message")

    execution_time_seconds: Optional[float] = Field(
        default=None,
        description="Total execution time in seconds"
    )

    output_files: Optional[Dict[str, str]] = Field(
        default=None,
        description="Generated output files by type"
    )

    summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pipeline execution summary"
    )

    errors: Optional[List[str]] = Field(
        default=None,
        description="Error messages if pipeline failed"
    )

    started_at: Optional[datetime] = Field(
        default=None,
        description="Job start timestamp"
    )

    completed_at: Optional[datetime] = Field(
        default=None,
        description="Job completion timestamp"
    )


class JobStatusResponse(BaseModel):
    """Response model for job status queries."""

    job_id: str = Field(description="Unique job identifier")
    status: JobStatus = Field(description="Current job status")
    progress: Optional[str] = Field(
        default=None,
        description="Current progress description"
    )

    started_at: Optional[datetime] = Field(
        default=None,
        description="Job start timestamp"
    )

    completed_at: Optional[datetime] = Field(
        default=None,
        description="Job completion timestamp"
    )

    execution_time_seconds: Optional[float] = Field(
        default=None,
        description="Total execution time in seconds"
    )

    result: Optional[PipelineResponse] = Field(
        default=None,
        description="Full pipeline result (only when completed)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
