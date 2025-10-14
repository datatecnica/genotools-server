"""
Minimal data loaders for frontend with caching.
"""

import os
import json
import pandas as pd
import streamlit as st
from typing import Optional, Dict, List


@st.cache_data(ttl=300)
def discover_releases(results_base_path: str) -> List[str]:
    """
    Discover available releases in results directory.

    Args:
        results_base_path: Base path to results directory (e.g., .../release10)

    Returns:
        List of release names sorted by most recent first

    Note:
        Cache expires after 5 minutes to allow automatic detection of new releases
    """
    if not os.path.exists(results_base_path):
        return []

    base_path = os.path.dirname(results_base_path)
    releases = []

    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and item.startswith('release'):
                releases.append(item)

    return sorted(releases, reverse=True)


@st.cache_data
def discover_jobs(release: str, results_base_path: str) -> List[str]:
    """
    Discover available job names within a release.

    Args:
        release: Release identifier (e.g., 'release10')
        results_base_path: Base path to results directory

    Returns:
        List of job names
    """
    release_path = os.path.join(os.path.dirname(results_base_path), release)
    if not os.path.exists(release_path):
        return []

    jobs = set()
    data_types = ['NBA', 'WGS', 'IMPUTED']

    # Extract job names from parquet files
    for file in os.listdir(release_path):
        if file.endswith('.parquet'):
            parts = file.replace('.parquet', '').split('_')
            if len(parts) >= 2 and parts[-1] in data_types:
                job_name = '_'.join(parts[:-1])
                jobs.add(job_name)
        elif file.endswith('_pipeline_results.json'):
            job_name = file.replace('_pipeline_results.json', '')
            jobs.add(job_name)

    # Add release name as default
    if not jobs:
        jobs.add(release)
    elif release not in jobs:
        jobs.add(release)

    return sorted(list(jobs))


@st.cache_data
def load_pipeline_results(release: str, job_name: str, results_base_path: str) -> Optional[Dict]:
    """
    Load pipeline results JSON file.

    Args:
        release: Release identifier
        job_name: Job name
        results_base_path: Base path to results directory

    Returns:
        Pipeline results dict or None if not found
    """
    release_path = os.path.join(os.path.dirname(results_base_path), release)
    file_path = os.path.join(release_path, f"{job_name}_pipeline_results.json")

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading pipeline results: {e}")
    return None


@st.cache_data
def load_locus_report(release: str, job_name: str, comparison: str, results_base_path: str) -> Optional[Dict]:
    """
    Load locus report JSON file.

    Args:
        release: Release identifier
        job_name: Job name
        comparison: Comparison type ('WGS_NBA' or 'WGS_IMPUTED')
        results_base_path: Base path to results directory

    Returns:
        Locus report dict or None if not found
    """
    release_path = os.path.join(os.path.dirname(results_base_path), release)
    file_path = os.path.join(release_path, f"{job_name}_locus_reports_{comparison}.json")

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading locus report: {e}")
    return None


@st.cache_data
def load_probe_validation(release: str, job_name: str, results_base_path: str) -> Optional[Dict]:
    """
    Load probe validation/selection JSON file.

    Args:
        release: Release identifier
        job_name: Job name
        results_base_path: Base path to results directory

    Returns:
        Probe validation dict or None if not found
    """
    release_path = os.path.join(os.path.dirname(results_base_path), release)
    file_path = os.path.join(release_path, f"{job_name}_probe_selection.json")

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading probe validation: {e}")
    return None


@st.cache_data
def check_data_availability(release: str, job_name: str, results_base_path: str) -> Dict[str, bool]:
    """
    Check which data files are available for a release/job.

    Args:
        release: Release identifier
        job_name: Job name
        results_base_path: Base path to results directory

    Returns:
        Dict with availability flags for each data type
    """
    release_path = os.path.join(os.path.dirname(results_base_path), release)

    return {
        'pipeline_results': os.path.exists(os.path.join(release_path, f"{job_name}_pipeline_results.json")),
        'locus_reports_nba': os.path.exists(os.path.join(release_path, f"{job_name}_locus_reports_NBA.json")),
        'locus_reports_imputed': os.path.exists(os.path.join(release_path, f"{job_name}_locus_reports_IMPUTED.json")),
        'probe_validation': os.path.exists(os.path.join(release_path, f"{job_name}_probe_selection.json"))
    }


@st.cache_data
def get_selected_probe_ids(release: str, job_name: str, results_base_path: str) -> Dict[str, any]:
    """
    Extract selected probe variant IDs from probe selection report.

    Args:
        release: Release identifier
        job_name: Job name
        results_base_path: Base path to results directory

    Returns:
        Dict with:
            - 'selected_ids': set of recommended variant IDs
            - 'total_mutations': total mutations analyzed
            - 'has_probe_selection': whether probe selection data exists
    """
    probe_data = load_probe_validation(release, job_name, results_base_path)

    if not probe_data:
        return {
            'selected_ids': set(),
            'total_mutations': 0,
            'has_probe_selection': False
        }

    selected_ids = set()
    for comparison in probe_data.get('probe_comparisons', []):
        consensus = comparison.get('consensus', {})
        recommended = consensus.get('recommended_probe')
        if recommended:
            selected_ids.add(recommended)

    summary = probe_data.get('summary', {})
    total_mutations_analyzed = summary.get('total_mutations_analyzed', 0)
    mutations_with_multiple_probes = summary.get('mutations_with_multiple_probes', 0)

    return {
        'selected_ids': selected_ids,
        'total_mutations': total_mutations_analyzed,
        'mutations_with_multiple_probes': mutations_with_multiple_probes,
        'has_probe_selection': True
    }


@st.cache_data
def load_variant_to_mutation_map(release: str, job_name: str, results_base_path: str) -> Dict[str, str]:
    """
    Load variant_id to mutation name (snp_list_id) mapping from available parquet files.

    This allows displaying human-readable mutation names alongside variant IDs.
    Tries WGS first (most comprehensive), then NBA, then IMPUTED.

    Args:
        release: Release identifier
        job_name: Job name
        results_base_path: Base path to results directory

    Returns:
        Dict mapping variant_id -> mutation name (e.g., 'chr1:7965425:G:C' -> 'E64D')
        Returns empty dict if no parquet files exist
    """
    release_path = os.path.join(os.path.dirname(results_base_path), release)
    variant_map = {}

    # Try each data type in order of preference
    data_types = ['WGS', 'NBA', 'IMPUTED']

    for data_type in data_types:
        parquet_path = os.path.join(release_path, f"{job_name}_{data_type}.parquet")

        if not os.path.exists(parquet_path):
            continue

        try:
            # Load only the columns we need for the mapping
            df = pd.read_parquet(parquet_path, columns=['variant_id', 'snp_list_id'])

            # Create variant_id -> mutation name mapping
            # Handle cases where variant_id might appear multiple times (different ancestries)
            for _, row in df[['variant_id', 'snp_list_id']].drop_duplicates().iterrows():
                variant_id = row['variant_id']
                mutation_name = row['snp_list_id']
                if pd.notna(mutation_name) and mutation_name and variant_id not in variant_map:
                    variant_map[variant_id] = mutation_name

        except Exception as e:
            # Continue to next data type if this one fails
            continue

    return variant_map


@st.cache_data
def load_variant_carrier_counts(release: str, job_name: str, results_base_path: str, data_type: str = 'WGS') -> Dict[str, int]:
    """
    Load carrier counts per variant from parquet files.

    Args:
        release: Release identifier
        job_name: Job name
        results_base_path: Base path to results directory
        data_type: Data type to load ('WGS', 'NBA', or 'IMPUTED')

    Returns:
        Dict mapping variant_id -> carrier count (number of samples with genotype > 0)
        Returns empty dict if parquet doesn't exist
    """
    release_path = os.path.join(os.path.dirname(results_base_path), release)
    parquet_path = os.path.join(release_path, f"{job_name}_{data_type}.parquet")

    if not os.path.exists(parquet_path):
        return {}

    try:
        # Load parquet
        df = pd.read_parquet(parquet_path)

        # Identify metadata vs sample columns
        metadata_cols = ['chromosome', 'variant_id', '(C)M', 'position', 'COUNTED', 'ALT',
                        'counted_allele', 'alt_allele', 'harmonization_action', 'snp_list_id',
                        'pgen_a1', 'pgen_a2', 'data_type', 'source_file', 'ancestry']
        sample_cols = [col for col in df.columns if col not in metadata_cols]

        # Calculate carrier counts for each variant
        carrier_counts = {}
        for _, row in df.iterrows():
            variant_id = row['variant_id']

            # Get genotypes and convert to numeric
            genotypes = pd.to_numeric(row[sample_cols], errors='coerce')

            # Count carriers (genotype > 0)
            carrier_count = (genotypes > 0).sum()
            carrier_counts[variant_id] = int(carrier_count)

        return carrier_counts

    except Exception as e:
        st.warning(f"Could not load carrier counts: {e}")
        return {}
