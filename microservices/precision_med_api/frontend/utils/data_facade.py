"""
Data facade providing simplified interface for complex data operations.
"""

from typing import List
from frontend.config import FrontendConfig
from frontend.models.frontend_models import OverviewData
from frontend.utils.data_loaders import DataLoaderFactory


class DataFacade:
    """Simplified interface for complex data operations."""

    def __init__(self, config: FrontendConfig):
        """Initialize facade with configuration."""
        self.config = config
        self.factory = DataLoaderFactory()

    def discover_releases(self) -> List[str]:
        """Discover available releases in results directory."""
        loader = self.factory.get_loader('releases')
        return loader.load(_config=self.config)

    def discover_jobs(self, release: str) -> List[str]:
        """Discover available jobs for a release."""
        loader = self.factory.get_loader('jobs')
        return loader.load(release, _config=self.config)

    def get_overview_data(self, release: str, job_name: str) -> OverviewData:
        """
        Orchestrate loading of all overview data.

        Args:
            release: Release identifier (e.g., 'release10')
            job_name: Job name identifier

        Returns:
            OverviewData: Consolidated overview data
        """
        # Load all required data using factory loaders
        pipeline_results = self.factory.get_loader('pipeline_results').load(
            release, job_name, _config=self.config
        )

        sample_counts = self.factory.get_loader('sample_counts').load(
            release, job_name, _config=self.config
        )

        file_info = self.factory.get_loader('file_info').load(
            release, job_name, config=self.config
        )

        return OverviewData(
            pipeline_results=pipeline_results,
            sample_counts=sample_counts,
            file_info=file_info
        )

    def validate_release_and_job(self, release: str, job_name: str) -> bool:
        """
        Validate that release and job exist and have data.

        Args:
            release: Release identifier
            job_name: Job name identifier

        Returns:
            bool: True if valid, False otherwise
        """
        # Check if release exists
        releases = self.discover_releases()
        if release not in releases:
            return False

        # Check if job exists in release
        jobs = self.discover_jobs(release)
        if job_name not in jobs:
            return False

        # Check if any data files exist
        file_info = self.factory.get_loader('file_info').load(
            release, job_name, config=self.config
        )
        return len(file_info) > 0

    def get_available_data_types(self, release: str, job_name: str) -> List[str]:
        """
        Get list of data types available for a release/job combination.

        Args:
            release: Release identifier
            job_name: Job name identifier

        Returns:
            List[str]: Available data types (NBA, WGS, IMPUTED)
        """
        file_info = self.factory.get_loader('file_info').load(
            release, job_name, config=self.config
        )
        return list(file_info.keys())