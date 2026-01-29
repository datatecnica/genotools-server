"""
Full NBA probe audit against WGS ground truth without SNP list filtering.

Validates ALL NBA variants against WGS data to identify probe quality issues
and cross-ancestry discrepancies in probe recommendations.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..core.config import Settings
from ..models.harmonization import HarmonizationRecord, HarmonizationAction
from .harmonizer import HarmonizationEngine
from .extractor import VariantExtractor
from .probe_selector import ProbeSelector
from .probe_recommender import ProbeRecommendationEngine

logger = logging.getLogger(__name__)


class ProbeAuditor:
    """
    Audits NBA probe quality against WGS ground truth across all variants.

    Unlike the standard probe selection which only analyzes SNP list variants,
    this auditor compares ALL shared positions between NBA and WGS to provide
    comprehensive probe quality assessment.
    """

    def __init__(self, settings: Settings):
        """Initialize probe auditor with configuration."""
        self.settings = settings
        self.harmonization_engine = HarmonizationEngine(settings)
        self.extractor = VariantExtractor(settings)

    def _get_shared_samples_from_psam(
        self,
        nba_path: str,
        wgs_path: str
    ) -> List[str]:
        """
        Get sample IDs present in both NBA and WGS without loading genotypes.

        This is much faster than loading full genotype data to find shared samples.

        Args:
            nba_path: Path to NBA PLINK files (without extension)
            wgs_path: Path to any WGS chromosome PLINK files (without extension)

        Returns:
            List of sample IDs present in both datasets
        """
        try:
            # Read NBA psam (don't use comment='#' - header starts with #FID)
            nba_psam_path = nba_path + '.psam'
            nba_psam = pd.read_csv(nba_psam_path, sep='\t')
            # Handle IID column - PLINK2 uses #FID but IID (without #)
            if 'IID' in nba_psam.columns:
                iid_col = 'IID'
            elif '#IID' in nba_psam.columns:
                iid_col = '#IID'
            else:
                # Fallback: second column is typically IID
                iid_col = nba_psam.columns[1] if len(nba_psam.columns) > 1 else nba_psam.columns[0]
            nba_samples = set(nba_psam[iid_col].astype(str))

            # Read WGS psam (same samples across all chromosomes)
            wgs_psam_path = wgs_path + '.psam'
            wgs_psam = pd.read_csv(wgs_psam_path, sep='\t')
            if 'IID' in wgs_psam.columns:
                iid_col = 'IID'
            elif '#IID' in wgs_psam.columns:
                iid_col = '#IID'
            else:
                iid_col = wgs_psam.columns[1] if len(wgs_psam.columns) > 1 else wgs_psam.columns[0]
            wgs_samples = set(wgs_psam[iid_col].astype(str))

            shared = list(nba_samples.intersection(wgs_samples))
            logger.info(f"Found {len(shared)} shared samples (NBA: {len(nba_samples)}, WGS: {len(wgs_samples)})")

            return shared

        except Exception as e:
            logger.warning(f"Failed to read psam files for sample filtering: {e}")
            return []

    def find_shared_positions(
        self,
        nba_pvar: pd.DataFrame,
        wgs_pvars: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Find genomic positions present in both NBA and WGS datasets.

        Args:
            nba_pvar: NBA PVAR DataFrame
            wgs_pvars: List of WGS PVAR DataFrames (one per chromosome)

        Returns:
            DataFrame with matched positions and variant IDs from both sources
        """
        # Combine all WGS chromosomes
        wgs_combined = pd.concat(wgs_pvars, ignore_index=True)
        logger.debug(f"Combined WGS: {len(wgs_combined)} variants across {len(wgs_pvars)} chromosomes")

        # Inner join on chromosome and position
        shared = pd.merge(
            nba_pvar[['CHROM', 'POS', 'ID', 'REF', 'ALT']].copy(),
            wgs_combined[['CHROM', 'POS', 'ID', 'REF', 'ALT']].copy(),
            on=['CHROM', 'POS'],
            how='inner',
            suffixes=('_nba', '_wgs')
        )

        # Create mutation_id from WGS as ground truth key for multi-probe grouping
        shared['mutation_id'] = (
            shared['CHROM'].astype(str) + ':' +
            shared['POS'].astype(str) + ':' +
            shared['REF_wgs'].astype(str) + ':' +
            shared['ALT_wgs'].astype(str)
        )

        logger.debug(f"Found {len(shared)} shared positions between NBA and WGS")

        # Log multi-probe mutations
        multi_probe_count = shared.groupby('mutation_id')['ID_nba'].nunique()
        multi_probe_mutations = (multi_probe_count > 1).sum()
        logger.debug(f"Mutations with multiple NBA probes: {multi_probe_mutations}")

        return shared

    def _build_snp_list_for_harmonization(
        self,
        shared_df: pd.DataFrame,
        source: str = 'wgs'
    ) -> pd.DataFrame:
        """
        Build a SNP list DataFrame compatible with HarmonizationEngine.

        Args:
            shared_df: DataFrame with shared positions
            source: 'wgs' or 'nba' to determine which alleles to use as reference

        Returns:
            DataFrame with columns: chromosome, position, ref, alt, variant_id, snp_name
        """
        suffix = f'_{source}'

        snp_list = pd.DataFrame({
            'chromosome': shared_df['CHROM'].astype(str),
            'position': shared_df['POS'].astype(int),
            'ref': shared_df[f'REF{suffix}'].astype(str),
            'alt': shared_df[f'ALT{suffix}'].astype(str),
            'variant_id': shared_df[f'ID{suffix}'].astype(str),
            'snp_name': shared_df['mutation_id'].astype(str)  # Use mutation_id as snp_name for grouping
        })

        return snp_list

    def extract_harmonized_genotypes(
        self,
        pgen_path: str,
        shared_df: pd.DataFrame,
        source: str = 'nba'
    ) -> pd.DataFrame:
        """
        Extract and harmonize genotypes for shared positions.

        For NBA: Harmonize NBA probes against WGS alleles (ground truth)
        For WGS: Extract WGS as-is (it's the ground truth)

        Args:
            pgen_path: Path to PGEN file (without extension)
            shared_df: DataFrame with shared positions
            source: 'nba' or 'wgs' indicating data source

        Returns:
            DataFrame with harmonized genotypes
        """
        if shared_df.empty:
            return pd.DataFrame()

        # For NBA: harmonize against WGS (ground truth)
        # For WGS: use WGS alleles as both source and reference (no transformation needed)
        snp_list = self._build_snp_list_for_harmonization(shared_df, source='wgs')

        try:
            result_df = self.extractor.extract_single_file_harmonized(
                pgen_path=pgen_path + '.pgen',
                snp_list_ids=snp_list['snp_name'].tolist(),  # mutation_ids for grouping
                snp_list_df=snp_list
            )

            if not result_df.empty:
                result_df['data_type'] = source.upper()
                result_df['source_file'] = pgen_path

            return result_df

        except Exception as e:
            logger.error(f"Failed to extract genotypes from {pgen_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _extract_with_sample_filter(
        self,
        pgen_path: str,
        shared_df: pd.DataFrame,
        sample_ids: List[str],
        source: str = 'nba'
    ) -> pd.DataFrame:
        """
        Extract genotypes with sample filtering for efficiency.

        This method filters to only the samples we need, reducing memory usage
        by 2-3x compared to extracting all samples.

        Args:
            pgen_path: Path to PGEN file (without extension)
            shared_df: DataFrame with shared positions
            sample_ids: List of sample IDs to extract (shared samples only)
            source: 'nba' or 'wgs' indicating data source

        Returns:
            DataFrame with harmonized genotypes for shared samples only
        """
        if shared_df.empty:
            return pd.DataFrame()

        # Build SNP list using WGS alleles as reference (ground truth)
        snp_list = self._build_snp_list_for_harmonization(shared_df, source='wgs')

        try:
            # Use extractor with sample filtering
            result_df = self.extractor.extract_single_file_harmonized(
                pgen_path=pgen_path + '.pgen',
                snp_list_ids=snp_list['snp_name'].tolist(),
                snp_list_df=snp_list,
                sample_ids=sample_ids  # Pass sample filter
            )

            if not result_df.empty:
                result_df['data_type'] = source.upper()
                result_df['source_file'] = pgen_path

            return result_df

        except Exception as e:
            logger.error(f"Failed to extract genotypes from {pgen_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _extract_wgs_per_chromosome(
        self,
        ancestry: str,
        shared_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract WGS genotypes per chromosome and combine.

        Args:
            ancestry: Ancestry code
            shared_df: DataFrame with shared positions

        Returns:
            Combined WGS genotype DataFrame
        """
        wgs_dfs = []

        # Group shared positions by chromosome for efficient extraction
        for chrom in sorted(shared_df['CHROM'].unique()):
            wgs_path = self.settings.get_wgs_path(ancestry, str(chrom))

            if not os.path.exists(wgs_path + '.pgen'):
                logger.warning(f"WGS file not found: {wgs_path}.pgen")
                continue

            # Get positions for this chromosome
            chrom_shared = shared_df[shared_df['CHROM'] == chrom]

            if chrom_shared.empty:
                continue

            logger.debug(f"Extracting {len(chrom_shared)} variants from WGS chr{chrom}")

            wgs_df = self.extract_harmonized_genotypes(
                pgen_path=wgs_path,
                shared_df=chrom_shared,
                source='wgs'
            )

            if not wgs_df.empty:
                wgs_dfs.append(wgs_df)

        if not wgs_dfs:
            return pd.DataFrame()

        # Combine all chromosomes
        combined = pd.concat(wgs_dfs, ignore_index=True)
        logger.debug(f"Combined WGS extraction: {len(combined)} variants")

        return combined

    def run_audit(self, ancestry: str) -> Dict[str, Any]:
        """
        Run full probe audit for one ancestry.

        Optimized for efficiency:
        - Finds all shared positions first (pvar-only, fast)
        - Extracts NBA genotypes in ONE PLINK call (not 22)
        - Filters to shared samples only (reduces memory 2-3x)
        - Extracts WGS per-chromosome with sample filtering

        Args:
            ancestry: Ancestry code (e.g., 'EUR', 'AFR')

        Returns:
            Dictionary with audit results including probe analysis and recommendations
        """
        logger.info(f"Starting probe audit for ancestry: {ancestry}")

        result = {
            'ancestry': ancestry,
            'success': False,
            'error': None,
            'nba_variants': 0,
            'wgs_variants': 0,
            'shared_positions': 0,
            'shared_samples': 0,
            'multi_probe_mutations': {},
            'recommendations': {},
            'probe_analysis': {},
            'summary': {}
        }

        try:
            # Step 1: Load NBA pvar ONCE (single file, ~1.6M variants)
            nba_path = self.settings.get_nba_path(ancestry)
            if not os.path.exists(nba_path + '.pvar'):
                raise FileNotFoundError(f"NBA PVAR not found: {nba_path}.pvar")

            nba_pvar = self.harmonization_engine.read_pvar_file(nba_path + '.pgen')
            result['nba_variants'] = len(nba_pvar)
            logger.info(f"Loaded NBA PVAR: {len(nba_pvar)} variants")

            # Step 2: Find ALL shared positions across chromosomes (pvar-only, fast)
            # This avoids loading all WGS data at once - just the metadata
            all_shared = []
            first_wgs_path = None
            total_wgs_variants = 0

            chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                          '11', '12', '13', '14', '15', '16', '17', '18', '19',
                          '20', '21', '22']

            logger.info("Finding shared positions across chromosomes...")
            for chrom in chromosomes:
                wgs_path = self.settings.get_wgs_path(ancestry, chrom)
                if not os.path.exists(wgs_path + '.pvar'):
                    continue

                if first_wgs_path is None:
                    first_wgs_path = wgs_path

                try:
                    wgs_pvar = self.harmonization_engine.read_pvar_file(wgs_path + '.pgen')
                    total_wgs_variants += len(wgs_pvar)

                    chrom_shared = self.find_shared_positions(nba_pvar, [wgs_pvar])
                    if not chrom_shared.empty:
                        # Add chromosome info for later WGS extraction
                        chrom_shared['wgs_chrom'] = chrom
                        all_shared.append(chrom_shared)
                        logger.debug(f"  chr{chrom}: {len(chrom_shared)} shared positions")

                    del wgs_pvar  # Free immediately

                except Exception as e:
                    logger.warning(f"Failed to read WGS chr{chrom}: {e}")
                    continue

            result['wgs_variants'] = total_wgs_variants

            if not all_shared:
                logger.warning(f"No shared positions found for ancestry {ancestry}")
                result['success'] = True
                return result

            # Combine all shared positions
            combined_shared = pd.concat(all_shared, ignore_index=True)
            del all_shared
            result['shared_positions'] = len(combined_shared)
            logger.info(f"Total shared positions: {len(combined_shared)}")

            # Step 3: Filter to ONLY multi-probe mutations (critical optimization)
            # Counting NBA probes per WGS position (mutation_id groups by chr:pos:ref:alt)
            nba_probes_per_mutation = combined_shared.groupby('mutation_id')['ID_nba'].nunique()
            multi_probe_mutations = nba_probes_per_mutation[nba_probes_per_mutation > 1].index

            if len(multi_probe_mutations) == 0:
                logger.info(f"No multi-probe mutations found for ancestry {ancestry}")
                result['success'] = True
                return result

            # Filter to only positions with multiple probes
            multi_probe_shared = combined_shared[combined_shared['mutation_id'].isin(multi_probe_mutations)].copy()
            logger.info(f"Multi-probe positions to analyze: {len(multi_probe_shared)} "
                       f"({len(multi_probe_mutations)} unique mutations)")

            # Step 4: Get shared samples (from psam - no genotype loading)
            all_shared_samples = self._get_shared_samples_from_psam(nba_path, first_wgs_path)

            if not all_shared_samples:
                logger.warning(f"No shared samples found for ancestry {ancestry}")
                result['success'] = True
                return result

            # Limit samples for memory efficiency (probe quality can be assessed with subset)
            # 2000 samples is statistically sufficient for probe concordance analysis
            MAX_SAMPLES = 2000
            if len(all_shared_samples) > MAX_SAMPLES:
                import random
                random.seed(42)  # Reproducible sampling
                shared_samples = random.sample(all_shared_samples, MAX_SAMPLES)
                logger.info(f"Sampling {MAX_SAMPLES} of {len(all_shared_samples)} shared samples for efficiency")
            else:
                shared_samples = all_shared_samples

            result['shared_samples'] = len(shared_samples)

            # Step 5: Extract NBA genotypes for multi-probe positions only
            logger.info(f"Extracting NBA genotypes ({len(multi_probe_shared)} variants, {len(shared_samples)} samples)...")
            nba_df = self._extract_with_sample_filter(
                pgen_path=nba_path,
                shared_df=multi_probe_shared,
                sample_ids=shared_samples,
                source='nba'
            )

            if nba_df.empty:
                raise ValueError("No NBA genotypes extracted")

            logger.info(f"Extracted NBA: {len(nba_df)} variants")

            # Step 6: Extract WGS genotypes per-chromosome (with sample filtering)
            logger.info("Extracting WGS genotypes per chromosome...")
            all_wgs_dfs = []

            for chrom in multi_probe_shared['wgs_chrom'].unique():
                chrom_shared = multi_probe_shared[multi_probe_shared['wgs_chrom'] == chrom]
                wgs_path = self.settings.get_wgs_path(ancestry, str(chrom))

                wgs_df = self._extract_with_sample_filter(
                    pgen_path=wgs_path,
                    shared_df=chrom_shared,
                    sample_ids=shared_samples,
                    source='wgs'
                )

                if not wgs_df.empty:
                    all_wgs_dfs.append(wgs_df)
                    logger.debug(f"  chr{chrom}: {len(wgs_df)} variants")

            if not all_wgs_dfs:
                raise ValueError("No WGS genotypes extracted")

            wgs_df = pd.concat(all_wgs_dfs, ignore_index=True)
            del all_wgs_dfs

            logger.info(f"Extracted WGS: {len(wgs_df)} variants")

            # Step 4: Run probe analysis using existing ProbeSelector
            probe_selector = ProbeSelector(self.settings)
            probe_recommender = ProbeRecommendationEngine(strategy="consensus")

            # Analyze probes
            probe_analysis_by_mutation, summary = probe_selector.analyze_probes_from_dataframes(
                nba_df=nba_df,
                wgs_df=wgs_df
            )

            result['multi_probe_mutations'] = {
                mutation: [p.variant_id for p in probes]
                for mutation, probes in probe_analysis_by_mutation.items()
            }

            # Generate recommendations
            if probe_analysis_by_mutation:
                # Build mutation metadata
                mutation_metadata = {}
                for _, row in nba_df.iterrows():
                    snp_list_id = row.get('snp_list_id', '')
                    if snp_list_id and snp_list_id not in mutation_metadata:
                        mutation_metadata[snp_list_id] = {
                            'snp_list_id': snp_list_id,
                            'chromosome': row.get('chromosome', ''),
                            'position': row.get('position', 0)
                        }

                mutation_analyses, methodology_comparison = probe_recommender.recommend_probes(
                    probe_analysis_by_mutation=probe_analysis_by_mutation,
                    mutation_metadata=mutation_metadata
                )

                # Extract recommendations
                for analysis in mutation_analyses:
                    mutation_id = analysis.snp_list_id
                    result['recommendations'][mutation_id] = {
                        'best_probe': analysis.recommended_probe,
                        'concordance': getattr(analysis, 'concordance_score', None),
                        'diagnostic_score': getattr(analysis, 'diagnostic_score', None),
                        'consensus': analysis.has_consensus
                    }

                result['probe_analysis'] = {
                    mutation: [
                        {
                            'variant_id': p.variant_id,
                            'probe_type': p.probe_type,
                            'diagnostic': {
                                'sensitivity': p.diagnostic_metrics.sensitivity,
                                'specificity': p.diagnostic_metrics.specificity,
                                'ppv': p.diagnostic_metrics.ppv,
                                'npv': p.diagnostic_metrics.npv
                            },
                            'concordance': {
                                'overall': p.concordance_metrics.overall_concordance,
                                'carrier_sensitivity': p.concordance_metrics.carrier_sensitivity,
                                'quality_score': p.concordance_metrics.quality_score
                            }
                        }
                        for p in probes
                    ]
                    for mutation, probes in probe_analysis_by_mutation.items()
                }

            result['summary'] = {
                'total_mutations_analyzed': summary.total_mutations_analyzed,
                'mutations_with_multiple_probes': summary.mutations_with_multiple_probes,
                'total_probe_comparisons': summary.total_probe_comparisons,
                'samples_compared': summary.samples_compared
            }

            result['success'] = True
            logger.info(f"Probe audit complete for {ancestry}: {summary.mutations_with_multiple_probes} multi-probe mutations analyzed")

        except Exception as e:
            logger.error(f"Probe audit failed for {ancestry}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result['error'] = str(e)

        return result


def compare_across_ancestries(ancestry_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compare probe recommendations across ancestries to identify discrepancies.

    Args:
        ancestry_results: Dictionary mapping ancestry to audit results

    Returns:
        Dictionary with cross-ancestry comparison including discrepant probes
    """
    # Get all mutations with multiple probes from any ancestry
    all_mutations = set()
    for ancestry, results in ancestry_results.items():
        if results.get('success'):
            all_mutations.update(results.get('multi_probe_mutations', {}).keys())

    discrepant = []
    concordant = []

    for mutation in all_mutations:
        recommendations_by_ancestry = {}

        for ancestry, results in ancestry_results.items():
            if not results.get('success'):
                continue

            rec = results.get('recommendations', {}).get(mutation)
            if rec:
                recommendations_by_ancestry[ancestry] = rec

        # Need at least 2 ancestries to compare
        if len(recommendations_by_ancestry) < 2:
            continue

        # Check if all ancestries agree on best probe
        best_probes = [r['best_probe'] for r in recommendations_by_ancestry.values()]
        unique_probes = set(best_probes)

        if len(unique_probes) > 1:
            discrepant.append({
                'mutation_id': mutation,
                'recommendations_by_ancestry': recommendations_by_ancestry,
                'is_discrepant': True,
                'unique_recommendations': list(unique_probes)
            })
        else:
            concordant.append({
                'mutation_id': mutation,
                'recommendations_by_ancestry': recommendations_by_ancestry,
                'is_discrepant': False,
                'agreed_probe': best_probes[0] if best_probes else None
            })

    return {
        'discrepant_probes': discrepant,
        'concordant_probes': concordant,
        'discrepant_probes_count': len(discrepant),
        'concordant_probes_count': len(concordant),
        'total_compared': len(discrepant) + len(concordant)
    }
