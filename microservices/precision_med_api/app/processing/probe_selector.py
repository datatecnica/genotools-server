"""
Probe quality analysis for multiple NBA probes per genomic position.

Analyzes NBA probe performance against WGS ground truth data using both
diagnostic classification and genotype concordance approaches.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..models.probe_validation import (
    DiagnosticMetrics,
    ConcordanceMetrics,
    GenotypeTransitionMatrix,
    ProbeAnalysisResult,
    ProbeSelectionSummary
)
from ..core.config import Settings

logger = logging.getLogger(__name__)


class ProbeSelector:
    """
    Analyzes NBA probe quality against WGS ground truth data.

    Generates comprehensive validation metrics using both diagnostic
    classification and genotype concordance approaches for probe
    quality assessment.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize probe selector with configuration."""
        self.settings = settings or Settings()

    def analyze_probes(
        self,
        nba_parquet_path: str,
        wgs_parquet_path: str
    ) -> Tuple[Dict[str, List[ProbeAnalysisResult]], ProbeSelectionSummary]:
        """
        Analyze NBA probe quality against WGS ground truth data.

        Args:
            nba_parquet_path: Path to NBA parquet file
            wgs_parquet_path: Path to WGS parquet file

        Returns:
            Tuple of (probe_analysis_by_mutation, summary_statistics)
        """
        logger.info("Loading NBA and WGS data for probe analysis")

        # Load data
        nba_df = pd.read_parquet(nba_parquet_path)
        wgs_df = pd.read_parquet(wgs_parquet_path)

        logger.info(f"Loaded NBA data: {nba_df.shape[0]} variants, {nba_df.shape[1]-15} samples")
        logger.info(f"Loaded WGS data: {wgs_df.shape[0]} variants, {wgs_df.shape[1]-15} samples")

        # Find mutations with multiple NBA probes
        mutations_with_multiple_probes = self._identify_multiple_probe_mutations(nba_df)
        logger.info(f"Found {len(mutations_with_multiple_probes)} mutations with multiple NBA probes")

        # Get shared samples between NBA and WGS
        shared_samples = self._get_shared_samples(nba_df, wgs_df)
        logger.info(f"Found {len(shared_samples)} shared samples between NBA and WGS")

        if len(shared_samples) == 0:
            raise ValueError("No shared samples found between NBA and WGS data")

        # Analyze probes for each mutation
        probe_analysis_results = {}
        total_probe_comparisons = 0

        for mutation, nba_variants in mutations_with_multiple_probes.items():
            logger.info(f"Analyzing {len(nba_variants)} probes for mutation: {mutation}")

            # Find corresponding WGS variant
            wgs_variant = self._find_matching_wgs_variant(mutation, wgs_df)
            if wgs_variant is None:
                logger.warning(f"No matching WGS variant found for mutation: {mutation}")
                continue

            # Analyze each NBA probe for this mutation
            probe_results = []
            for nba_variant_id in nba_variants:
                result = self._analyze_single_probe(
                    nba_variant_id, wgs_variant, nba_df, wgs_df, shared_samples
                )
                probe_results.append(result)
                total_probe_comparisons += 1

            probe_analysis_results[mutation] = probe_results

        # Generate summary statistics
        summary = ProbeSelectionSummary(
            total_mutations_analyzed=len(probe_analysis_results),
            mutations_with_multiple_probes=len(mutations_with_multiple_probes),
            total_probe_comparisons=total_probe_comparisons,
            samples_compared=len(shared_samples)
        )

        logger.info(f"Probe analysis complete: {summary.total_mutations_analyzed} mutations analyzed")
        return probe_analysis_results, summary

    def _identify_multiple_probe_mutations(self, nba_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify mutations with multiple NBA probes.

        Args:
            nba_df: NBA dataframe with variant metadata

        Returns:
            Dictionary mapping snp_list_id to list of variant_ids
        """
        # Group by snp_list_id to find mutations with multiple probes
        mutations_with_multiple_probes = {}

        for _, row in nba_df.iterrows():
            snp_list_id = row['snp_list_id']
            variant_id = row['variant_id']

            if snp_list_id not in mutations_with_multiple_probes:
                mutations_with_multiple_probes[snp_list_id] = []
            mutations_with_multiple_probes[snp_list_id].append(variant_id)

        # Filter to only mutations with multiple probes
        return {
            mutation: variants
            for mutation, variants in mutations_with_multiple_probes.items()
            if len(variants) > 1
        }

    def _get_shared_samples(self, nba_df: pd.DataFrame, wgs_df: pd.DataFrame) -> List[str]:
        """
        Get sample IDs that exist in both NBA and WGS datasets.

        Args:
            nba_df: NBA dataframe
            wgs_df: WGS dataframe

        Returns:
            List of shared sample IDs
        """
        # Get sample columns (excluding metadata columns)
        metadata_cols = ['variant_id', 'snp_list_id', 'chromosome', 'position',
                        'original_a1', 'original_a2', 'counted_allele', 'alt_allele',
                        'harmonization_action', 'data_type', 'source_file']

        nba_samples = set(col for col in nba_df.columns if col not in metadata_cols)
        wgs_samples = set(col for col in wgs_df.columns if col not in metadata_cols)

        return list(nba_samples.intersection(wgs_samples))

    def _find_matching_wgs_variant(self, snp_list_id: str, wgs_df: pd.DataFrame) -> Optional[str]:
        """
        Find WGS variant that matches the given snp_list_id.

        Args:
            snp_list_id: SNP list identifier
            wgs_df: WGS dataframe

        Returns:
            WGS variant_id if found, None otherwise
        """
        matching_rows = wgs_df[wgs_df['snp_list_id'] == snp_list_id]
        if len(matching_rows) == 0:
            return None
        elif len(matching_rows) == 1:
            return matching_rows.iloc[0]['variant_id']
        else:
            # Multiple WGS variants for same mutation - take first one
            logger.warning(f"Multiple WGS variants found for {snp_list_id}, using first one")
            return matching_rows.iloc[0]['variant_id']

    def _analyze_single_probe(
        self,
        nba_variant_id: str,
        wgs_variant_id: str,
        nba_df: pd.DataFrame,
        wgs_df: pd.DataFrame,
        shared_samples: List[str]
    ) -> ProbeAnalysisResult:
        """
        Analyze a single NBA probe against WGS ground truth.

        Args:
            nba_variant_id: NBA variant identifier
            wgs_variant_id: WGS variant identifier
            nba_df: NBA dataframe
            wgs_df: WGS dataframe
            shared_samples: List of shared sample IDs

        Returns:
            ProbeAnalysisResult with both diagnostic and concordance metrics
        """
        # Get genotype data for shared samples
        nba_row = nba_df[nba_df['variant_id'] == nba_variant_id].iloc[0]
        wgs_row = wgs_df[wgs_df['variant_id'] == wgs_variant_id].iloc[0]

        nba_genotypes = nba_row[shared_samples].values
        wgs_genotypes = wgs_row[shared_samples].values

        # Clean data - convert to numeric and handle missing values
        nba_genotypes = pd.to_numeric(nba_genotypes, errors='coerce')
        wgs_genotypes = pd.to_numeric(wgs_genotypes, errors='coerce')

        # Handle NaN values
        nba_genotypes = np.nan_to_num(nba_genotypes, nan=0.0).astype(int)
        wgs_genotypes = np.nan_to_num(wgs_genotypes, nan=0.0).astype(int)

        # Calculate diagnostic metrics (carrier vs non-carrier)
        diagnostic_metrics = self._calculate_diagnostic_metrics(nba_genotypes, wgs_genotypes)

        # Calculate concordance metrics (genotype-level agreement)
        concordance_metrics = self._calculate_concordance_metrics(nba_genotypes, wgs_genotypes)

        return ProbeAnalysisResult(
            variant_id=nba_variant_id,
            probe_type=self._infer_probe_type(nba_variant_id),
            diagnostic_metrics=diagnostic_metrics,
            concordance_metrics=concordance_metrics
        )

    def _calculate_diagnostic_metrics(
        self,
        nba_genotypes: np.ndarray,
        wgs_genotypes: np.ndarray
    ) -> DiagnosticMetrics:
        """
        Calculate diagnostic test metrics treating carriers as positive cases.

        Args:
            nba_genotypes: NBA genotype array (0/1/2)
            wgs_genotypes: WGS genotype array (0/1/2)

        Returns:
            DiagnosticMetrics with TP/FP/FN/TN counts
        """
        # Convert to binary: 0 = non-carrier, 1|2 = carrier
        nba_carriers = (nba_genotypes > 0).astype(int)
        wgs_carriers = (wgs_genotypes > 0).astype(int)

        # Calculate confusion matrix
        tp = np.sum((nba_carriers == 1) & (wgs_carriers == 1))
        fp = np.sum((nba_carriers == 1) & (wgs_carriers == 0))
        fn = np.sum((nba_carriers == 0) & (wgs_carriers == 1))
        tn = np.sum((nba_carriers == 0) & (wgs_carriers == 0))

        return DiagnosticMetrics(
            true_positives=int(tp),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_negatives=int(tn)
        )

    def _calculate_concordance_metrics(
        self,
        nba_genotypes: np.ndarray,
        wgs_genotypes: np.ndarray
    ) -> ConcordanceMetrics:
        """
        Calculate genotype concordance metrics with transition matrix.

        Args:
            nba_genotypes: NBA genotype array (0/1/2)
            wgs_genotypes: WGS genotype array (0/1/2)

        Returns:
            ConcordanceMetrics with transition matrix and derived statistics
        """
        # Build transition matrix
        transition_counts = {}
        for nba_gt in [0, 1, 2]:
            for wgs_gt in [0, 1, 2]:
                count = np.sum((nba_genotypes == nba_gt) & (wgs_genotypes == wgs_gt))
                transition_counts[f"nba_{nba_gt}_wgs_{wgs_gt}"] = int(count)

        transition_matrix = GenotypeTransitionMatrix(**transition_counts)

        return ConcordanceMetrics(
            total_samples_compared=len(nba_genotypes),
            transition_matrix=transition_matrix
        )

    def _infer_probe_type(self, variant_id: str) -> str:
        """
        Infer probe type from variant identifier.

        Args:
            variant_id: Variant identifier string

        Returns:
            Probe type classification
        """
        variant_id_lower = variant_id.lower()

        if 'seq_' in variant_id_lower:
            return "sequencing"
        elif 'ilmn' in variant_id_lower:
            return "illumina_array"
        elif 'chr' in variant_id_lower and ':' in variant_id_lower:
            return "coordinate_based"
        elif 'nm_' in variant_id_lower or 'hgvs' in variant_id_lower:
            return "transcript_based"
        else:
            return "unknown"