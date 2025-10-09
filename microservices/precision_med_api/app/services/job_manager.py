"""
Job tracking and management for long-running pipeline executions.
"""

import uuid
import logging
from typing import Dict, Optional
from datetime import datetime
from threading import Lock

from ..models.api import JobStatus, PipelineResponse, JobStatusResponse

logger = logging.getLogger(__name__)


class JobManager:
    """
    Thread-safe in-memory job tracking system.

    Tracks pipeline execution status, progress, and results.
    For production use, consider replacing with Redis or database backend.
    """

    def __init__(self):
        self._jobs: Dict[str, Dict] = {}
        self._lock = Lock()

    def create_job(self, job_name: str, request_params: Dict) -> str:
        """
        Create a new job entry.

        Args:
            job_name: Human-readable job name
            request_params: Request parameters for reproducibility

        Returns:
            Unique job ID
        """
        job_id = str(uuid.uuid4())

        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "job_name": job_name,
                "status": JobStatus.PENDING,
                "progress": "Job created, waiting to start",
                "started_at": None,
                "completed_at": None,
                "execution_time_seconds": None,
                "request_params": request_params,
                "result": None,
                "errors": []
            }

        logger.info(f"Created job {job_id} ({job_name})")
        return job_id

    def start_job(self, job_id: str):
        """Mark job as running."""
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")

            self._jobs[job_id]["status"] = JobStatus.RUNNING
            self._jobs[job_id]["started_at"] = datetime.utcnow()
            self._jobs[job_id]["progress"] = "Pipeline execution started"

        logger.info(f"Started job {job_id}")

    def update_progress(self, job_id: str, progress: str):
        """Update job progress message."""
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")

            self._jobs[job_id]["progress"] = progress

        logger.debug(f"Job {job_id} progress: {progress}")

    def complete_job(self, job_id: str, result: PipelineResponse):
        """Mark job as completed with results."""
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")

            job = self._jobs[job_id]
            job["status"] = JobStatus.COMPLETED
            job["completed_at"] = datetime.utcnow()
            job["result"] = result
            job["progress"] = "Pipeline completed successfully"

            # Calculate execution time
            if job["started_at"]:
                delta = job["completed_at"] - job["started_at"]
                job["execution_time_seconds"] = delta.total_seconds()

        logger.info(f"Completed job {job_id}")

    def fail_job(self, job_id: str, errors: list):
        """Mark job as failed with error messages."""
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")

            job = self._jobs[job_id]
            job["status"] = JobStatus.FAILED
            job["completed_at"] = datetime.utcnow()
            job["errors"] = errors
            job["progress"] = "Pipeline failed"

            # Calculate execution time
            if job["started_at"]:
                delta = job["completed_at"] - job["started_at"]
                job["execution_time_seconds"] = delta.total_seconds()

        logger.error(f"Failed job {job_id}: {errors}")

    def get_job_status(self, job_id: str) -> Optional[JobStatusResponse]:
        """
        Get current job status.

        Args:
            job_id: Job identifier

        Returns:
            JobStatusResponse or None if not found
        """
        with self._lock:
            if job_id not in self._jobs:
                return None

            job = self._jobs[job_id]

            return JobStatusResponse(
                job_id=job["job_id"],
                status=job["status"],
                progress=job["progress"],
                started_at=job["started_at"],
                completed_at=job["completed_at"],
                execution_time_seconds=job["execution_time_seconds"],
                result=job["result"]
            )

    def get_job_result(self, job_id: str) -> Optional[PipelineResponse]:
        """
        Get job result if completed.

        Args:
            job_id: Job identifier

        Returns:
            PipelineResponse or None if not found/not completed
        """
        with self._lock:
            if job_id not in self._jobs:
                return None

            job = self._jobs[job_id]

            if job["status"] != JobStatus.COMPLETED:
                return None

            return job["result"]

    def job_exists(self, job_id: str) -> bool:
        """Check if job exists."""
        with self._lock:
            return job_id in self._jobs

    def list_jobs(self, limit: int = 100) -> list:
        """List recent jobs (for debugging/monitoring)."""
        with self._lock:
            jobs = list(self._jobs.values())
            # Sort by creation time (most recent first)
            jobs.sort(
                key=lambda x: x.get("started_at") or datetime.min,
                reverse=True
            )
            return jobs[:limit]


# Global singleton instance
_job_manager = JobManager()


def get_job_manager() -> JobManager:
    """Get global job manager instance."""
    return _job_manager
