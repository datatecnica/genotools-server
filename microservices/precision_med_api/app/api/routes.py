"""
API routes for carriers pipeline endpoints.
"""

import logging
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from datetime import datetime

from ..models.api import (
    PipelineRequest,
    PipelineResponse,
    JobStatusResponse,
    JobStatus,
    HealthResponse
)
from ..services.job_manager import get_job_manager
from ..services.pipeline_service import PipelineService

logger = logging.getLogger(__name__)

router = APIRouter()


def run_pipeline_background(job_id: str, request: PipelineRequest):
    """
    Background task for executing pipeline.

    Args:
        job_id: Job identifier for tracking
        request: Pipeline execution parameters
    """
    job_manager = get_job_manager()
    pipeline_service = PipelineService()

    try:
        # Start job
        job_manager.start_job(job_id)

        # Execute pipeline
        logger.info(f"Starting pipeline execution for job {job_id}")
        results = pipeline_service.execute_pipeline(request)

        # Create response
        response = PipelineResponse(
            success=results['success'],
            job_id=results['job_id'],
            status=JobStatus.COMPLETED if results['success'] else JobStatus.FAILED,
            message="Pipeline completed successfully" if results['success'] else "Pipeline failed",
            execution_time_seconds=results.get('execution_time_seconds'),
            output_files=results.get('output_files'),
            summary=results.get('summary'),
            errors=results.get('errors'),
            started_at=job_manager.get_job_status(job_id).started_at,
            completed_at=datetime.utcnow()
        )

        # Update job status
        if results['success']:
            job_manager.complete_job(job_id, response)
        else:
            job_manager.fail_job(job_id, results.get('errors', ['Unknown error']))

    except Exception as e:
        logger.error(f"Pipeline execution failed for job {job_id}: {e}")
        import traceback
        error_trace = traceback.format_exc()
        job_manager.fail_job(job_id, [str(e), error_trace])


@router.post("/pipeline", response_model=PipelineResponse)
async def start_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks
):
    """
    Start carriers pipeline execution.

    Initiates a background job for pipeline execution and returns
    immediately with job tracking information.

    Args:
        request: Pipeline configuration parameters
        background_tasks: FastAPI background tasks manager

    Returns:
        PipelineResponse with job ID for tracking
    """
    job_manager = get_job_manager()

    # Create job
    job_id = job_manager.create_job(
        job_name=request.job_name,
        request_params=request.model_dump()
    )

    # Schedule background execution
    background_tasks.add_task(run_pipeline_background, job_id, request)

    logger.info(f"Pipeline job {job_id} created and scheduled")

    # Return immediate response
    return PipelineResponse(
        success=True,
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Pipeline job created. Use /pipeline/{job_id} to check status.",
        started_at=None,
        completed_at=None
    )


@router.get("/pipeline/{job_id}", response_model=JobStatusResponse)
async def get_pipeline_status(job_id: str):
    """
    Get pipeline job status and results.

    Args:
        job_id: Job identifier

    Returns:
        JobStatusResponse with current status and results (if completed)

    Raises:
        HTTPException: If job not found (404)
    """
    job_manager = get_job_manager()

    status = job_manager.get_job_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return status


@router.get("/pipeline/{job_id}/results", response_model=PipelineResponse)
async def get_pipeline_results(job_id: str):
    """
    Get completed pipeline results.

    Args:
        job_id: Job identifier

    Returns:
        PipelineResponse with execution results

    Raises:
        HTTPException: If job not found (404) or not completed (400)
    """
    job_manager = get_job_manager()

    if not job_manager.job_exists(job_id):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result = job_manager.get_job_result(job_id)

    if result is None:
        status = job_manager.get_job_status(job_id)
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed yet. Current status: {status.status}"
        )

    return result


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with service status
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )
