"""
Data loaders using factory pattern.
"""

import os
import json
import pandas as pd
import streamlit as st
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from frontend.config import FrontendConfig

logger = logging.getLogger(__name__)


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


class LocusReportsLoader(DataLoader):
    """Loader for locus reports."""

    @st.cache_data(ttl=300)
    def load(
        _self,
        config: 'FrontendConfig',
        release: str,
        job_name: str,
        comparison: str = "WGS_NBA"
    ) -> Optional[Dict[str, Any]]:
        """
        Load locus report data for a specific comparison.

        Args:
            config: Frontend configuration
            release: Release identifier
            job_name: Job name
            comparison: Data type comparison ("WGS_NBA" or "WGS_IMPUTED")

        Returns:
            Locus report collection data or None if not available
        """
        try:
            release_path = os.path.join(os.path.dirname(config.results_base_path), release)
            report_path = os.path.join(release_path, f"{job_name}_locus_reports_{comparison}.json")

            if not os.path.exists(report_path):
                return None

            with open(report_path, 'r') as f:
                data = json.load(f)

            return data

        except Exception as e:
            logger.error(f"Failed to load locus report data: {e}")
            return None


class GenotypeDataLoader(DataLoader):
    """Loader for genotype data from parquet files with subsetting capabilities."""

    @st.cache_data(ttl=600)
    def load(_self, release: str, job_name: str, _config: FrontendConfig,
             data_types: Optional[List[str]] = None, genes: Optional[List[str]] = None,
             carrier_only: bool = False) -> Dict[str, Any]:
        """
        Load genotype data with optional filtering.

        Args:
            release: Release identifier
            job_name: Job name
            _config: Frontend configuration
            data_types: List of data types to load (NBA, WGS, IMPUTED). If None, loads all available.
            genes: List of genes/loci to filter to. If None, loads all.
            carrier_only: If True, only return samples with at least one carrier genotype (>0)

        Returns:
            Dictionary with genotype data, metadata, and summary statistics
        """
        if not _config:
            return {"error": "Configuration not provided"}

        release_path = os.path.join(os.path.dirname(_config.results_base_path), release)
        if not os.path.exists(release_path):
            return {"error": f"Release path not found: {release_path}"}

        # Define metadata columns to exclude from genotype matrix
        metadata_cols = {
            'variant_id', 'snp_list_id', 'chromosome', 'position',
            'counted_allele', 'alt_allele', 'harmonization_action',
            'data_type', 'ancestry', 'pgen_variant_id', 'snp_list_a1',
            'snp_list_a2', 'pgen_a1', 'pgen_a2', 'file_path', 'source_file',
            'genotype_transform', 'original_a1', 'original_a2', '(C)M',
            'COUNTED', 'ALT', 'rsid', 'locus', 'snp_name'
        }

        # Use all available data types if none specified
        if data_types is None:
            data_types = _config.data_types

        loaded_data = {}
        summary_stats = {
            'total_variants': 0,
            'total_samples': 0,
            'total_carriers': 0,
            'data_types_loaded': [],
            'genes_available': [],
            'carrier_frequency': 0.0
        }

        try:
            for data_type in data_types:
                file_path = os.path.join(release_path, f"{job_name}_{data_type}.parquet")

                if not os.path.exists(file_path):
                    continue

                # Load parquet file
                df = pd.read_parquet(file_path)

                if df.empty:
                    continue

                # Filter by genes if specified
                if genes:
                    if 'locus' in df.columns:
                        df = df[df['locus'].isin(genes)]
                    elif 'snp_name' in df.columns:
                        # Filter by SNP names that contain gene names
                        gene_mask = df['snp_name'].str.contains('|'.join(genes), case=False, na=False)
                        df = df[gene_mask]

                if df.empty:
                    continue

                # Separate metadata and sample columns
                metadata_columns = [col for col in df.columns if col in metadata_cols]
                sample_columns = [col for col in df.columns if col not in metadata_cols]

                if not sample_columns:
                    continue

                # Get metadata
                metadata_df = df[metadata_columns].copy() if metadata_columns else pd.DataFrame(index=df.index)

                # Get genotype matrix
                genotype_df = df[sample_columns].copy()

                # Convert to numeric, handling missing values
                genotype_df = genotype_df.apply(pd.to_numeric, errors='coerce')

                # Filter to carriers only if requested
                if carrier_only:
                    # Find variants that have at least one carrier (genotype > 0)
                    carrier_variants = (genotype_df > 0).any(axis=1)
                    if carrier_variants.any():
                        genotype_df = genotype_df[carrier_variants]
                        metadata_df = metadata_df[carrier_variants]
                    else:
                        continue

                # Calculate statistics
                variant_count = len(genotype_df)
                sample_count = len(sample_columns)
                carrier_count = (genotype_df > 0).sum().sum()

                # Store data
                loaded_data[data_type] = {
                    'metadata': metadata_df,
                    'genotypes': genotype_df,
                    'sample_columns': sample_columns,
                    'stats': {
                        'variants': variant_count,
                        'samples': sample_count,
                        'carriers': carrier_count,
                        'carrier_frequency': carrier_count / (variant_count * sample_count) if variant_count * sample_count > 0 else 0.0
                    }
                }

                # Update summary statistics
                summary_stats['total_variants'] += variant_count
                summary_stats['total_samples'] = max(summary_stats['total_samples'], sample_count)
                summary_stats['total_carriers'] += carrier_count
                summary_stats['data_types_loaded'].append(data_type)

                # Extract available genes
                if 'locus' in metadata_df.columns:
                    available_genes = metadata_df['locus'].dropna().unique().tolist()
                    summary_stats['genes_available'].extend(available_genes)

            # Calculate overall carrier frequency
            if summary_stats['total_variants'] > 0 and summary_stats['total_samples'] > 0:
                total_possible_genotypes = summary_stats['total_variants'] * summary_stats['total_samples']
                summary_stats['carrier_frequency'] = summary_stats['total_carriers'] / total_possible_genotypes

            # Remove duplicates from genes
            summary_stats['genes_available'] = sorted(list(set(summary_stats['genes_available'])))

            return {
                'data': loaded_data,
                'summary': summary_stats,
                'filters_applied': {
                    'data_types': data_types,
                    'genes': genes,
                    'carrier_only': carrier_only
                }
            }

        except Exception as e:
            return {"error": f"Failed to load genotype data: {str(e)}"}

    @st.cache_data(ttl=600)
    def get_available_genes(_self, release: str, job_name: str, _config: FrontendConfig) -> List[str]:
        """
        Get list of available genes/loci from the data.

        Args:
            release: Release identifier
            job_name: Job name
            _config: Frontend configuration

        Returns:
            List of available gene names
        """
        if not _config:
            return []

        release_path = os.path.join(os.path.dirname(_config.results_base_path), release)
        genes = set()

        try:
            for data_type in _config.data_types:
                file_path = os.path.join(release_path, f"{job_name}_{data_type}.parquet")

                if os.path.exists(file_path):
                    try:
                        # Check if locus column exists first
                        sample_df = pd.read_parquet(file_path).head(1)
                        if 'locus' in sample_df.columns:
                            df = pd.read_parquet(file_path, columns=['locus'])
                            available_genes = df['locus'].dropna().unique().tolist()
                            genes.update(available_genes)
                    except Exception as e:
                        logger.warning(f"Could not read genes from {file_path}: {e}")
                        continue

            return sorted(list(genes))

        except Exception:
            return []


class DataLoaderFactory:
    """Factory for creating data loaders."""

    _loaders = {
        'releases': ReleaseDiscoveryLoader,
        'jobs': JobDiscoveryLoader,
        'pipeline_results': PipelineResultsLoader,
        'sample_counts': SampleCountsLoader,
        'file_info': FileInfoLoader,
        'probe_validation': ProbeValidationLoader,
        'locus_reports': LocusReportsLoader,
        'genotype_data': GenotypeDataLoader
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