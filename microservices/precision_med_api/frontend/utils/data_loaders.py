"""
Data loaders using factory pattern.
"""

import os
import json
import pandas as pd
import streamlit as st
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from frontend.config import FrontendConfig


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, release: str, job_name: str, config: FrontendConfig) -> Any:
        """Load data for given release and job."""
        pass


class ReleaseDiscoveryLoader(DataLoader):
    """Loader for discovering available releases."""

    @st.cache_data
    def load(_self, release: str = None, job_name: str = None, _config: FrontendConfig = None) -> List[str]:
        """Discover available releases in results directory."""
        if not _config or not os.path.exists(_config.results_base_path):
            return []

        releases = []
        base_path = os.path.dirname(_config.results_base_path)  # Go up one level from release{X}

        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path) and item.startswith('release'):
                    releases.append(item)

        return sorted(releases, reverse=True)  # Most recent first


class JobDiscoveryLoader(DataLoader):
    """Loader for discovering available jobs within a release."""

    @st.cache_data
    def load(_self, release: str, job_name: str = None, _config: FrontendConfig = None) -> List[str]:
        """Discover available job names in a release directory."""
        if not _config:
            return []

        release_path = os.path.join(os.path.dirname(_config.results_base_path), release)
        if not os.path.exists(release_path):
            return []

        jobs = set()

        # Look for parquet files and extract job names
        for file in os.listdir(release_path):
            if file.endswith('.parquet'):
                # Format: {job_name}_{data_type}.parquet
                parts = file.replace('.parquet', '').split('_')
                if len(parts) >= 2 and parts[-1] in _config.data_types:
                    job_name = '_'.join(parts[:-1])
                    jobs.add(job_name)
            elif file.endswith('_pipeline_results.json'):
                # Format: {job_name}_pipeline_results.json
                job_name = file.replace('_pipeline_results.json', '')
                jobs.add(job_name)

        # Add release name as default if no specific jobs found
        if not jobs:
            jobs.add(release)
        elif release not in jobs:
            jobs.add(release)

        return sorted(list(jobs))


class PipelineResultsLoader(DataLoader):
    """Loader for pipeline results JSON files."""

    @st.cache_data
    def load(_self, release: str, job_name: str, _config: FrontendConfig) -> Optional[Dict]:
        """Load pipeline results JSON file."""
        release_path = os.path.join(os.path.dirname(_config.results_base_path), release)
        file_path = os.path.join(release_path, f"{job_name}_pipeline_results.json")

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading pipeline results: {e}")
        return None


class SampleCountsLoader(DataLoader):
    """Loader for calculating sample counts from parquet files."""

    @st.cache_data
    def load(_self, release: str, job_name: str, _config: FrontendConfig) -> Dict[str, int]:
        """Get sample counts from parquet files."""
        sample_counts = {}
        total_samples = 0
        release_path = os.path.join(os.path.dirname(_config.results_base_path), release)

        # Metadata columns to exclude from sample count
        metadata_cols = {
            'variant_id', 'snp_list_id', 'chromosome', 'position',
            'counted_allele', 'alt_allele', 'harmonization_action',
            'data_type', 'ancestry', 'pgen_variant_id', 'snp_list_a1',
            'snp_list_a2', 'pgen_a1', 'pgen_a2', 'file_path', 'source_file',
            'genotype_transform', 'original_a1', 'original_a2', '(C)M',
            'COUNTED', 'ALT'
        }

        for data_type in _config.data_types:
            file_path = os.path.join(release_path, f"{job_name}_{data_type}.parquet")
            if os.path.exists(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    sample_cols = [col for col in df.columns if col not in metadata_cols]
                    sample_count = len(sample_cols)
                    sample_counts[data_type] = sample_count
                    total_samples += sample_count
                except Exception as e:
                    st.error(f"Error counting samples in {data_type}: {e}")

        sample_counts['TOTAL'] = total_samples
        return sample_counts


class FileInfoLoader(DataLoader):
    """Loader for file information and sizes."""

    def load(self, release: str, job_name: str, config: FrontendConfig) -> Dict[str, Dict[str, Any]]:
        """Get information about available files for a release."""
        release_path = os.path.join(os.path.dirname(config.results_base_path), release)
        file_info = {}

        for data_type in config.data_types:
            files = {}

            # Check for parquet and variant summary files
            for ext, desc in [('.parquet', 'Genotype Data'), ('_variant_summary.csv', 'Variant Summary')]:
                if ext == '_variant_summary.csv':
                    file_path = os.path.join(release_path, f"{job_name}_{data_type}{ext}")
                else:
                    file_path = os.path.join(release_path, f"{job_name}_{data_type}{ext}")

                if os.path.exists(file_path):
                    stat = os.stat(file_path)
                    size_bytes = stat.st_size

                    # Better size formatting for small files
                    if size_bytes < 1024:
                        size_display = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        size_display = f"{size_bytes / 1024:.1f} KB"
                    else:
                        size_display = f"{size_bytes / (1024 * 1024):.1f} MB"

                    files[desc] = {
                        'path': file_path,
                        'size_mb': round(stat.st_size / 1024 / 1024, 1),
                        'size_display': size_display,
                        'size_bytes': size_bytes,
                        'exists': True
                    }

            if files:  # Only add data type if files exist
                file_info[data_type] = files

        # Check for probe selection report
        probe_report_path = os.path.join(release_path, f"{job_name}_probe_selection.json")
        if os.path.exists(probe_report_path):
            stat = os.stat(probe_report_path)
            size_bytes = stat.st_size

            if size_bytes < 1024:
                size_display = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_display = f"{size_bytes / 1024:.1f} KB"
            else:
                size_display = f"{size_bytes / (1024 * 1024):.1f} MB"

            file_info["PROBE_VALIDATION"] = {
                "Probe Selection Report": {
                    "path": probe_report_path,
                    "size_mb": size_bytes / (1024 * 1024),
                    "size_display": size_display
                }
            }

        return file_info


class ProbeValidationLoader(DataLoader):
    """Loader for probe validation reports."""

    @st.cache_data(ttl=300)
    def load(_self, config: 'FrontendConfig', release: str, job_name: str) -> Optional[Dict[str, Any]]:
        """
        Load probe validation report data.

        Args:
            config: Frontend configuration
            release: Release identifier
            job_name: Job name

        Returns:
            Probe validation report data or None if not available
        """
        try:
            release_path = os.path.join(os.path.dirname(config.results_base_path), release)
            probe_report_path = os.path.join(release_path, f"{job_name}_probe_selection.json")

            if not os.path.exists(probe_report_path):
                return None

            with open(probe_report_path, 'r') as f:
                data = json.load(f)

            return data

        except Exception as e:
            logger.error(f"Failed to load probe validation data: {e}")
            return None


class DataLoaderFactory:
    """Factory for creating data loaders."""

    _loaders = {
        'releases': ReleaseDiscoveryLoader,
        'jobs': JobDiscoveryLoader,
        'pipeline_results': PipelineResultsLoader,
        'sample_counts': SampleCountsLoader,
        'file_info': FileInfoLoader,
        'probe_validation': ProbeValidationLoader
    }

    @classmethod
    def get_loader(cls, loader_type: str) -> DataLoader:
        """Get a data loader instance by type."""
        if loader_type not in cls._loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")
        return cls._loaders[loader_type]()

    @classmethod
    def list_available_loaders(cls) -> List[str]:
        """Get list of available loader types."""
        return list(cls._loaders.keys())