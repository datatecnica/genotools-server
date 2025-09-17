"""
Frontend-specific data models.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class OverviewData:
    """Data container for overview section."""

    pipeline_results: Optional[Dict[str, Any]]
    sample_counts: Dict[str, int]
    file_info: Dict[str, Dict[str, Any]]

    @property
    def total_variants(self) -> int:
        """Get total number of variants across all data types."""
        if self.pipeline_results and 'summary' in self.pipeline_results:
            return self.pipeline_results['summary'].get('total_variants', 0)
        return 0

    @property
    def variants_by_data_type(self) -> Dict[str, int]:
        """Get number of variants for each data type."""
        if self.pipeline_results and 'summary' in self.pipeline_results:
            by_data_type = self.pipeline_results['summary'].get('by_data_type', {})
            return {
                'NBA': by_data_type.get('NBA', {}).get('variants', 0),
                'WGS': by_data_type.get('WGS', {}).get('variants', 0),
                'IMPUTED': by_data_type.get('IMPUTED', {}).get('variants', 0)
            }
        return {'NBA': 0, 'WGS': 0, 'IMPUTED': 0}

    @property
    def pipeline_success(self) -> bool:
        """Check if pipeline execution was successful."""
        if self.pipeline_results:
            return self.pipeline_results.get('success', False)
        return False

    @property
    def execution_time(self) -> Optional[float]:
        """Get pipeline execution time in seconds."""
        if self.pipeline_results:
            return self.pipeline_results.get('execution_time_seconds')
        return None

    @property
    def start_time(self) -> Optional[str]:
        """Get pipeline start time."""
        if self.pipeline_results:
            start_time = self.pipeline_results.get('start_time', '')
            if start_time:
                return start_time.replace('T', ' ').split('.')[0]
        return None


    @property
    def error_count(self) -> int:
        """Get number of errors during pipeline execution."""
        if self.pipeline_results and 'errors' in self.pipeline_results:
            errors = self.pipeline_results['errors']
            return len(errors) if errors else 0
        return 0