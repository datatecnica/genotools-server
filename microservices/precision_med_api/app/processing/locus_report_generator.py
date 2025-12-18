"""
Locus report generator for clinical phenotype analysis.

Generates ancestry-stratified clinical phenotype tables for each gene/locus
by joining genotype data with clinical data sources.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

from app.core.config import Settings
from app.models.locus_report import (
    ClinicalMetrics,
    LocusReport,
    LocusReportSummary,
    LocusReportCollection
)
from app.models.analysis import DataType
from app.utils.probe_selector_loader import ProbeSelectionLoader


class LocusReportGenerator:
    """Generates per-locus clinical phenotype reports from extraction results."""

    def __init__(self, settings: Settings, probe_selection_path: Optional[str] = None):
        """Initialize with configuration settings.

        Args:
            settings: Application configuration settings
            probe_selection_path: Optional path to probe selection JSON file
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # Load reference data
        self.master_key = self._load_master_key()
        self.extended_clinical = self._load_extended_clinical()
        self.snp_list = self._load_snp_list()

        # Load probe selection if provided
        self.probe_selector = ProbeSelectionLoader(probe_selection_path)
        if self.probe_selector.has_probe_selection():
            stats = self.probe_selector.get_statistics()
            self.logger.info(
                f"Probe selection enabled: {stats['mutations_with_selection']} mutations, "
                f"{stats['probes_excluded']} inferior probes will be filtered"
            )
        else:
            self.logger.info("Probe selection disabled or not available")

    def _load_master_key(self) -> pd.DataFrame:
        """Load master key file with ancestry labels."""
        key_path = Path(self.settings.release_path) / "clinical_data" / f"master_key_release{self.settings.release}_final_vwb.csv"
        self.logger.info(f"Loading master key from: {key_path}")

        df = pd.read_csv(key_path)
        self.logger.info(f"Loaded {len(df):,} samples from master key")

        # Select relevant columns (including age data for disease duration calculation)
        cols_to_keep = ['GP2ID', 'nba_label', 'nba', 'wgs', 'extended_clinical_data']

        # Add age columns if available (for disease duration calculation)
        if 'age_at_sample_collection' in df.columns:
            cols_to_keep.append('age_at_sample_collection')
        if 'age_of_onset' in df.columns:
            cols_to_keep.append('age_of_onset')

        df = df[cols_to_keep].copy()
        return df

    def _load_extended_clinical(self) -> pd.DataFrame:
        """Load extended clinical data file."""
        import glob

        clinical_dir = Path(self.settings.release_path) / "clinical_data"

        # Use glob to find extended clinical file (handles varying naming patterns across releases)
        # Patterns: r8_extended_clinical_data_vwb_2024-09-11.csv, r10_extended_clinical_data_vwb.csv, r11_extended_clinical_data.csv
        # Exclude data dictionary files (*_dictionary.csv)
        pattern = str(clinical_dir / f"r{self.settings.release}_extended_clinical_data*.csv")
        matches = glob.glob(pattern)

        # Filter out dictionary files
        matches = [m for m in matches if 'dictionary' not in m.lower()]

        if not matches:
            raise FileNotFoundError(f"No extended clinical file found matching pattern: {pattern}")

        clin_path = Path(matches[0])  # Use first match
        self.logger.info(f"Loading extended clinical from: {clin_path}")

        df = pd.read_csv(clin_path, low_memory=False)
        self.logger.info(f"Loaded {len(df):,} clinical records")

        # Filter to baseline visits only (visit_month == 0)
        baseline_df = df[df['visit_month'] == 0].copy()
        self.logger.info(f"Filtered to {len(baseline_df):,} baseline visits")

        # Select relevant columns (age data comes from master key, not here)
        clinical_cols = [
            'GP2ID',
            'Phenotype',
            'visit_month',
            'moca_total_score',
            'hoehn_and_yahr_stage',
            'dat_sbr_caudate_mean'
        ]

        # Only keep columns that exist
        available_cols = [col for col in clinical_cols if col in baseline_df.columns]
        baseline_df = baseline_df[available_cols].copy()

        return baseline_df

    def _load_snp_list(self) -> pd.DataFrame:
        """Load SNP list with locus annotations."""
        snp_path = Path(self.settings.snp_list_path)
        self.logger.info(f"Loading SNP list from: {snp_path}")

        df = pd.read_csv(snp_path)
        self.logger.info(f"Loaded {len(df):,} variants from SNP list")

        # Select relevant columns
        df = df[['snp_name', 'locus', 'hg38']].copy()
        df = df.rename(columns={'snp_name': 'snp_list_id'})

        return df

    def _filter_to_selected_probes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to include only selected probes for multi-probe mutations.

        For mutations with multiple probes, keeps only the recommended variant.
        For mutations with single probe, keeps that probe (default selected).
        For multi-probe mutations not in probe selection (no WGS comparison), keeps all probes.

        Args:
            df: Genotype DataFrame with variant_id and snp_list_id columns

        Returns:
            Filtered DataFrame with only selected probes
        """
        if df.empty:
            return df

        initial_count = len(df)
        self.logger.info(f"Applying probe selection filter: {initial_count} variants before filtering")

        # Group by snp_list_id to identify single vs multiple probe mutations
        snp_counts = df.groupby('snp_list_id').size()

        # Track mutations without probe selection
        mutations_without_selection = set()

        # Build filter mask
        keep_mask = pd.Series([False] * len(df), index=df.index)

        for idx, row in df.iterrows():
            snp_list_id = row['snp_list_id']
            variant_id = row['variant_id']
            probe_count = snp_counts.get(snp_list_id, 1)

            # Single probe mutation: keep (default selected)
            if probe_count == 1:
                keep_mask[idx] = True
            else:
                # Multiple probes: check if this is the recommended one
                recommended = self.probe_selector.get_recommended_variant(snp_list_id)
                if recommended:
                    # We have a recommendation, only keep the selected probe
                    if recommended == variant_id:
                        keep_mask[idx] = True
                else:
                    # No recommendation (mutation not in WGS for comparison)
                    # Keep all probes for this mutation (can't determine which is better)
                    keep_mask[idx] = True
                    mutations_without_selection.add(snp_list_id)

        # Apply filter
        filtered_df = df[keep_mask].copy()
        final_count = len(filtered_df)
        removed_count = initial_count - final_count

        # Log summary
        if mutations_without_selection:
            self.logger.info(
                f"Note: {len(mutations_without_selection)} multi-probe mutations not in probe selection "
                f"(no WGS comparison available), kept all probes for these mutations"
            )

        self.logger.info(
            f"Probe selection filtering complete: {final_count} variants kept, "
            f"{removed_count} inferior probes removed"
        )

        return filtered_df

    def generate_reports(
        self,
        parquet_files: Dict[str, str],
        output_dir: str,
        job_name: str,
        data_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Generate independent locus reports for each data type.

        Args:
            parquet_files: Dict mapping data type to parquet file path
            output_dir: Output directory for report files
            job_name: Job name for output file naming
            data_types: List of data types to generate reports for (e.g., ["WGS", "NBA", "IMPUTED"])
                       If None, generates for all available data types

        Returns:
            Dictionary mapping output file types to file paths
        """
        if data_types is None:
            # Generate for all available data types
            data_types = list(parquet_files.keys())

        output_files = {}

        for data_type in data_types:
            # Check if data type is available
            if data_type not in parquet_files:
                self.logger.warning(f"Skipping {data_type} - data not available")
                continue

            self.logger.info(f"\n=== Generating {data_type} Locus Reports ===")

            # Generate report for this data type
            collection = self._generate_datatype_report(
                data_path=parquet_files[data_type],
                data_type=data_type,
                job_name=job_name
            )

            # Save outputs
            json_path = Path(output_dir) / f"{job_name}_locus_reports_{data_type}.json"
            csv_path = Path(output_dir) / f"{job_name}_locus_reports_{data_type}.csv"

            # Save JSON
            with open(json_path, 'w') as f:
                f.write(collection.model_dump_json(indent=2))
            output_files[f"locus_reports_{data_type}_json"] = str(json_path)
            self.logger.info(f"Saved JSON report: {json_path}")

            # Save CSV (flattened table)
            csv_df = self._flatten_to_csv(collection)
            csv_df.to_csv(csv_path, index=False)
            output_files[f"locus_reports_{data_type}_csv"] = str(csv_path)
            self.logger.info(f"Saved CSV report: {csv_path}")

            # Log summary
            self.logger.info(f"Generated reports for {len(collection.locus_reports)} loci")
            self.logger.info(f"Total carriers identified: {collection.summary.total_carriers_identified:,}")
            self.logger.info(f"Total variants: {collection.summary.total_variants:,}")

        return output_files

    def _generate_datatype_report(
        self,
        data_path: str,
        data_type: str,
        job_name: str
    ) -> LocusReportCollection:
        """Generate locus report for a single data type (no merging).

        Args:
            data_path: Path to data type parquet
            data_type: Data type name (WGS, NBA, or IMPUTED)
            job_name: Job identifier

        Returns:
            LocusReportCollection with all locus reports for this data type
        """
        # Load genotype data
        self.logger.info(f"Loading {data_type} data from: {data_path}")
        df = pd.read_parquet(data_path)

        # Apply probe selection filtering (NBA only)
        if data_type == "NBA" and self.probe_selector.has_probe_selection():
            df = self._filter_to_selected_probes(df)

        # Calculate variant-level carrier counts
        variant_details_by_variant = self._calculate_variant_carrier_counts(df)

        # Join with clinical data (melts to long format)
        clinical_df = self._join_clinical_data(df)

        # Group by locus and generate reports
        locus_reports = self._calculate_locus_metrics_with_variants(clinical_df, variant_details_by_variant)

        # Calculate summary statistics
        summary = self._calculate_summary_with_totals(locus_reports, clinical_df, df)

        # Create collection
        collection = LocusReportCollection(
            job_id=job_name,
            analysis_timestamp=datetime.now(),
            data_type=data_type,
            summary=summary,
            locus_reports=locus_reports,
            clinical_data_sources={
                "master_key": str(Path(self.settings.release_path) / "clinical_data" / f"master_key_release{self.settings.release}_final_vwb.csv"),
                "extended_clinical": str(Path(self.settings.release_path) / "clinical_data" / f"r{self.settings.release}_extended_clinical_data_vwb.csv")
            }
        )

        return collection

    def _generate_comparison_report(
        self,
        ref_path: str,
        compare_path: str,
        ref_type: str,
        compare_type: str,
        job_name: str
    ) -> LocusReportCollection:
        """Generate locus report for one data type comparison.

        Args:
            ref_path: Path to reference data parquet (e.g., WGS)
            compare_path: Path to comparison data parquet (e.g., NBA or IMPUTED)
            ref_type: Reference data type name
            compare_type: Comparison data type name
            job_name: Job identifier

        Returns:
            LocusReportCollection with all locus reports
        """
        # Load genotype data
        self.logger.info(f"Loading {ref_type} data from: {ref_path}")
        ref_df = pd.read_parquet(ref_path)

        self.logger.info(f"Loading {compare_type} data from: {compare_path}")
        compare_df = pd.read_parquet(compare_path)

        # Merge genotype datasets
        merged_genotypes = self._merge_genotype_data(ref_df, compare_df, ref_type, compare_type)

        # Join with clinical data
        clinical_df = self._join_clinical_data(merged_genotypes)

        # Group by locus and generate reports
        locus_reports = self._calculate_locus_metrics(clinical_df)

        # Calculate summary statistics
        summary = self._calculate_summary(locus_reports, clinical_df)

        # Create collection (using data_type field with combined name for backward compatibility)
        collection = LocusReportCollection(
            job_id=job_name,
            analysis_timestamp=datetime.now(),
            data_type=f"{ref_type}+{compare_type}",
            summary=summary,
            locus_reports=locus_reports,
            clinical_data_sources={
                "master_key": str(Path(self.settings.release_path) / "clinical_data" / f"master_key_release{self.settings.release}_final_vwb.csv"),
                "extended_clinical": str(Path(self.settings.release_path) / "clinical_data" / f"r{self.settings.release}_extended_clinical_data_vwb.csv")
            }
        )

        return collection

    def _merge_genotype_data(
        self,
        ref_df: pd.DataFrame,
        compare_df: pd.DataFrame,
        ref_type: str,
        compare_type: str
    ) -> pd.DataFrame:
        """Merge reference and comparison genotype data."""
        self.logger.info(f"Merging {ref_type} and {compare_type} genotype data")

        # Get metadata columns and sample columns
        metadata_cols = ['variant_id', 'snp_list_id', 'chromosome', 'position', 'ancestry', 'data_type']
        ref_metadata = ref_df[[col for col in metadata_cols if col in ref_df.columns]].copy()
        compare_metadata = compare_df[[col for col in metadata_cols if col in compare_df.columns]].copy()

        # Get sample columns (all columns except metadata)
        ref_sample_cols = [col for col in ref_df.columns if col not in metadata_cols]
        compare_sample_cols = [col for col in compare_df.columns if col not in metadata_cols]

        # Find common samples (intersection)
        common_samples = sorted(set(ref_sample_cols) & set(compare_sample_cols))
        self.logger.info(f"Found {len(common_samples):,} common samples between {ref_type} and {compare_type}")

        # Combine metadata with common samples
        # Use reference (WGS) variant_id and snp_list_id as primary keys
        keep_cols = ['variant_id', 'snp_list_id'] + common_samples
        merged = ref_df[[col for col in keep_cols if col in ref_df.columns]].copy()

        return merged

    def _calculate_variant_carrier_counts(self, df: pd.DataFrame) -> Dict[str, 'VariantDetail']:
        """Calculate carrier counts for each variant.

        Args:
            df: Genotype dataframe with samples as columns

        Returns:
            Dict mapping variant_id -> VariantDetail with carrier counts
        """
        from app.models.locus_report import VariantDetail

        self.logger.info("Calculating per-variant carrier counts")

        # Identify metadata vs sample columns
        metadata_cols = ['chromosome', 'variant_id', '(C)M', 'position', 'COUNTED', 'ALT',
                        'counted_allele', 'alt_allele', 'harmonization_action', 'snp_list_id',
                        'pgen_a1', 'pgen_a2', 'data_type', 'source_file', 'ancestry']
        sample_cols = [col for col in df.columns if col not in metadata_cols]

        variant_details = {}

        for _, row in df.iterrows():
            variant_id = row['variant_id']
            mutation_name = row.get('snp_list_id', '')

            # Parse variant ID to get genomic coordinates
            # Handle different formats: chr1:7965425:G:C or 1:8025485-G-C
            chromosome = str(row.get('chromosome', ''))
            position = int(row.get('position', 0))

            # Try to parse ref/alt from variant_id if not in columns
            ref_allele = row.get('pgen_a2', '')  # Reference allele
            alt_allele = row.get('pgen_a1', '')  # Alternate allele

            # If still empty, try parsing from variant_id
            if not ref_allele or not alt_allele:
                parts = str(variant_id).replace('-', ':').split(':')
                if len(parts) >= 4:
                    ref_allele = parts[2] if not ref_allele else ref_allele
                    alt_allele = parts[3] if not alt_allele else alt_allele

            # Get genotypes and convert to numeric
            genotypes = pd.to_numeric(row[sample_cols], errors='coerce')

            # Count carriers using configurable thresholds
            # For discrete genotypes (0,1,2): defaults work correctly
            # For imputed dosages (0.0-2.0): thresholds categorize appropriately
            het_min = self.settings.dosage_het_min
            het_max = self.settings.dosage_het_max
            hom_min = self.settings.dosage_hom_min

            het_count = ((genotypes >= het_min) & (genotypes < het_max)).sum()
            hom_count = (genotypes >= hom_min).sum()
            carrier_count = het_count + hom_count

            variant_details[variant_id] = VariantDetail(
                variant_id=variant_id,
                mutation_name=mutation_name if pd.notna(mutation_name) else "",
                chromosome=chromosome,
                position=position,
                ref_allele=str(ref_allele) if pd.notna(ref_allele) else "",
                alt_allele=str(alt_allele) if pd.notna(alt_allele) else "",
                carrier_count=int(carrier_count),
                heterozygous_count=int(het_count),
                homozygous_count=int(hom_count)
            )

        self.logger.info(f"Calculated carrier counts for {len(variant_details)} variants")
        return variant_details

    def _join_clinical_data(self, genotype_df: pd.DataFrame) -> pd.DataFrame:
        """Join genotype data with clinical data."""
        self.logger.info("Joining genotype data with clinical data")

        # Identify metadata vs sample columns
        metadata_cols = ['chromosome', 'variant_id', '(C)M', 'position', 'COUNTED', 'ALT',
                        'counted_allele', 'alt_allele', 'harmonization_action', 'snp_list_id',
                        'pgen_a1', 'pgen_a2', 'data_type', 'source_file', 'ancestry']

        # Select only ID vars and sample columns (exclude other metadata)
        id_vars = [col for col in ['variant_id', 'snp_list_id'] if col in genotype_df.columns]
        sample_cols = [col for col in genotype_df.columns if col not in metadata_cols]

        # Melt genotypes to long format with only sample columns
        genotype_long = genotype_df[id_vars + sample_cols].melt(
            id_vars=id_vars,
            var_name='GP2ID',
            value_name='genotype'
        )

        # Convert genotype to numeric (handle string values)
        genotype_long['genotype'] = pd.to_numeric(genotype_long['genotype'], errors='coerce')

        # Filter to carriers only (genotype > 0)
        carriers = genotype_long[genotype_long['genotype'] > 0].copy()
        self.logger.info(f"Identified {len(carriers):,} carrier genotypes")

        # Join with SNP list to get locus
        carriers = carriers.merge(
            self.snp_list[['snp_list_id', 'locus']],
            on='snp_list_id',
            how='left'
        )

        # Join with master key to get ancestry and age data for disease duration
        master_key_cols = ['GP2ID', 'nba_label', 'extended_clinical_data']
        # Include age columns if available
        if 'age_at_sample_collection' in self.master_key.columns:
            master_key_cols.append('age_at_sample_collection')
        if 'age_of_onset' in self.master_key.columns:
            master_key_cols.append('age_of_onset')

        carriers = carriers.merge(
            self.master_key[master_key_cols],
            on='GP2ID',
            how='left'
        )

        # Rename ancestry column
        carriers = carriers.rename(columns={'nba_label': 'ancestry'})

        # Join with extended clinical data
        carriers = carriers.merge(
            self.extended_clinical,
            on='GP2ID',
            how='left'
        )

        self.logger.info(f"Joined clinical data for {len(carriers):,} carrier records")

        return carriers

    def _calculate_locus_metrics(self, clinical_df: pd.DataFrame) -> List[LocusReport]:
        """Calculate clinical metrics grouped by locus."""
        self.logger.info("Calculating per-locus metrics")

        locus_reports = []

        # Group by locus
        for locus, locus_df in clinical_df.groupby('locus'):
            if pd.isna(locus):
                continue

            # Get variant info
            n_variants = locus_df['variant_id'].nunique()
            variant_ids = locus_df['variant_id'].unique().tolist()

            # Calculate metrics by ancestry
            ancestry_metrics = []
            for ancestry, ancestry_df in locus_df.groupby('ancestry'):
                if pd.isna(ancestry):
                    continue

                metrics = self._calculate_ancestry_metrics(ancestry_df, ancestry)
                ancestry_metrics.append(metrics)

            # Calculate total metrics (across all ancestries)
            total_metrics = self._calculate_ancestry_metrics(locus_df, "TOTAL")

            # Create locus report
            report = LocusReport(
                locus=str(locus),
                by_ancestry=ancestry_metrics,
                total_metrics=total_metrics,
                n_variants=n_variants,
                variant_ids=variant_ids
            )

            locus_reports.append(report)

        self.logger.info(f"Generated reports for {len(locus_reports)} loci")

        return sorted(locus_reports, key=lambda x: x.locus)

    def _calculate_locus_metrics_with_variants(
        self,
        clinical_df: pd.DataFrame,
        variant_details_map: Dict[str, 'VariantDetail']
    ) -> List[LocusReport]:
        """Calculate clinical metrics grouped by locus, including variant details.

        Args:
            clinical_df: Clinical data joined with genotypes (long format)
            variant_details_map: Dict mapping variant_id -> VariantDetail

        Returns:
            List of LocusReport objects with variant details
        """
        self.logger.info("Calculating per-locus metrics with variant details")

        locus_reports = []

        # Group by locus
        for locus, locus_df in clinical_df.groupby('locus'):
            if pd.isna(locus):
                continue

            # Get variant info
            variant_ids = locus_df['variant_id'].unique().tolist()
            n_variants = len(variant_ids)

            # Get variant details for this locus
            variant_details_list = [variant_details_map[vid] for vid in variant_ids if vid in variant_details_map]

            # Calculate metrics by ancestry
            ancestry_metrics = []
            for ancestry, ancestry_df in locus_df.groupby('ancestry'):
                if pd.isna(ancestry):
                    continue

                metrics = self._calculate_ancestry_metrics(ancestry_df, ancestry)
                ancestry_metrics.append(metrics)

            # Calculate total metrics (across all ancestries)
            total_metrics = self._calculate_ancestry_metrics(locus_df, "TOTAL")

            # Create locus report with variant details
            report = LocusReport(
                locus=str(locus),
                by_ancestry=ancestry_metrics,
                total_metrics=total_metrics,
                variant_details=variant_details_list,
                n_variants=n_variants,
                variant_ids=variant_ids
            )

            locus_reports.append(report)

        self.logger.info(f"Generated reports for {len(locus_reports)} loci with variant details")

        return sorted(locus_reports, key=lambda x: x.locus)

    def _calculate_ancestry_metrics(self, df: pd.DataFrame, ancestry: str) -> ClinicalMetrics:
        """Calculate clinical metrics for one ancestry group."""
        # Get unique carriers (one row per GP2ID)
        unique_carriers = df.drop_duplicates(subset=['GP2ID'])

        total_carriers = len(unique_carriers)

        # Count carriers with extended clinical data
        carriers_with_clinical = unique_carriers['extended_clinical_data'].sum() if 'extended_clinical_data' in unique_carriers.columns else 0

        # H&Y stage metrics
        hy_available = unique_carriers['hoehn_and_yahr_stage'].notna().sum() if 'hoehn_and_yahr_stage' in unique_carriers.columns else 0
        hy_less_than_2 = 0
        hy_less_than_3 = 0
        if 'hoehn_and_yahr_stage' in unique_carriers.columns:
            hy_values = pd.to_numeric(unique_carriers['hoehn_and_yahr_stage'], errors='coerce')
            hy_less_than_2 = (hy_values < 2).sum()
            hy_less_than_3 = (hy_values < 3).sum()

        # MoCA metrics
        moca_available = unique_carriers['moca_total_score'].notna().sum() if 'moca_total_score' in unique_carriers.columns else 0
        moca_gte_20 = 0
        moca_gte_24 = 0
        if 'moca_total_score' in unique_carriers.columns:
            moca_values = pd.to_numeric(unique_carriers['moca_total_score'], errors='coerce')
            moca_gte_20 = (moca_values >= 20).sum()
            moca_gte_24 = (moca_values >= 24).sum()

        # DAT caudate metrics
        dat_caudate_available = unique_carriers['dat_sbr_caudate_mean'].notna().sum() if 'dat_sbr_caudate_mean' in unique_carriers.columns else 0

        # Disease duration metrics (using age_at_sample_collection - age_of_onset from master key)
        disease_duration_lte_3 = 0
        disease_duration_lte_5 = 0
        disease_duration_lte_7 = 0
        if 'age_at_sample_collection' in unique_carriers.columns and 'age_of_onset' in unique_carriers.columns:
            duration = pd.to_numeric(unique_carriers['age_at_sample_collection'], errors='coerce') - pd.to_numeric(unique_carriers['age_of_onset'], errors='coerce')
            # Only count valid durations (non-negative)
            valid_duration = duration[duration >= 0]
            disease_duration_lte_3 = (valid_duration <= 3).sum()
            disease_duration_lte_5 = (valid_duration <= 5).sum()
            disease_duration_lte_7 = (valid_duration <= 7).sum()

        return ClinicalMetrics(
            ancestry=ancestry,
            total_carriers=int(total_carriers),
            carriers_with_clinical_data=int(carriers_with_clinical),
            hy_available=int(hy_available),
            hy_less_than_2=int(hy_less_than_2),
            hy_less_than_3=int(hy_less_than_3),
            moca_available=int(moca_available),
            moca_gte_20=int(moca_gte_20),
            moca_gte_24=int(moca_gte_24),
            dat_caudate_available=int(dat_caudate_available),
            disease_duration_lte_3_years=int(disease_duration_lte_3),
            disease_duration_lte_5_years=int(disease_duration_lte_5),
            disease_duration_lte_7_years=int(disease_duration_lte_7)
        )

    def _calculate_summary(self, locus_reports: List[LocusReport], clinical_df: pd.DataFrame) -> LocusReportSummary:
        """Calculate summary statistics across all loci."""
        total_carriers = clinical_df['GP2ID'].nunique()
        total_with_clinical = clinical_df[clinical_df['extended_clinical_data'] == 1]['GP2ID'].nunique() if 'extended_clinical_data' in clinical_df.columns else 0

        # Get ancestry representation
        ancestries = clinical_df['ancestry'].dropna().unique().tolist()
        carriers_by_ancestry = clinical_df.groupby('ancestry')['GP2ID'].nunique().to_dict()

        return LocusReportSummary(
            total_loci_analyzed=len(locus_reports),
            total_carriers_identified=int(total_carriers),
            total_samples_with_clinical_data=int(total_with_clinical),
            ancestries_represented=sorted(ancestries),
            carriers_by_ancestry={str(k): int(v) for k, v in carriers_by_ancestry.items()}
        )

    def _calculate_summary_with_totals(
        self,
        locus_reports: List[LocusReport],
        clinical_df: pd.DataFrame,
        full_df: pd.DataFrame
    ) -> LocusReportSummary:
        """Calculate summary statistics including total samples and variants.

        Args:
            locus_reports: List of locus reports
            clinical_df: Clinical dataframe (carriers only, long format)
            full_df: Full genotype dataframe (all variants, wide format)

        Returns:
            LocusReportSummary with complete statistics
        """
        total_carriers = clinical_df['GP2ID'].nunique()
        total_with_clinical = clinical_df[clinical_df['extended_clinical_data'] == 1]['GP2ID'].nunique() if 'extended_clinical_data' in clinical_df.columns else 0

        # Get ancestry representation
        ancestries = clinical_df['ancestry'].dropna().unique().tolist()
        carriers_by_ancestry = clinical_df.groupby('ancestry')['GP2ID'].nunique().to_dict()

        # Calculate total samples and variants from full dataframe
        metadata_cols = ['chromosome', 'variant_id', '(C)M', 'position', 'COUNTED', 'ALT',
                        'counted_allele', 'alt_allele', 'harmonization_action', 'snp_list_id',
                        'pgen_a1', 'pgen_a2', 'data_type', 'source_file', 'ancestry']
        sample_cols = [col for col in full_df.columns if col not in metadata_cols]
        total_samples = len(sample_cols)
        total_variants = len(full_df)

        return LocusReportSummary(
            total_loci_analyzed=len(locus_reports),
            total_carriers_identified=int(total_carriers),
            total_samples_analyzed=int(total_samples),
            total_samples_with_clinical_data=int(total_with_clinical),
            total_variants=int(total_variants),
            ancestries_represented=sorted(ancestries),
            carriers_by_ancestry={str(k): int(v) for k, v in carriers_by_ancestry.items()}
        )

    def _flatten_to_csv(self, collection: LocusReportCollection) -> pd.DataFrame:
        """Flatten report collection to CSV table format."""
        rows = []

        for report in collection.locus_reports:
            # Add rows for each ancestry
            for metrics in report.by_ancestry:
                row = {
                    'locus': report.locus,
                    'data_type': collection.data_type,
                    'ancestry': metrics.ancestry,
                    'total_carriers': metrics.total_carriers,
                    'carriers_with_clinical_data': metrics.carriers_with_clinical_data,
                    'hy_available': metrics.hy_available,
                    'hy_less_than_2': metrics.hy_less_than_2,
                    'hy_less_than_3': metrics.hy_less_than_3,
                    'moca_available': metrics.moca_available,
                    'moca_gte_20': metrics.moca_gte_20,
                    'moca_gte_24': metrics.moca_gte_24,
                    'dat_caudate_available': metrics.dat_caudate_available,
                    'disease_duration_lte_3_years': metrics.disease_duration_lte_3_years,
                    'disease_duration_lte_5_years': metrics.disease_duration_lte_5_years,
                    'disease_duration_lte_7_years': metrics.disease_duration_lte_7_years,
                    'clinical_data_availability_pct': f"{metrics.clinical_data_availability_pct:.1f}%"
                }
                rows.append(row)

            # Add total row
            total = report.total_metrics
            row = {
                'locus': report.locus,
                'data_type': collection.data_type,
                'ancestry': total.ancestry,
                'total_carriers': total.total_carriers,
                'carriers_with_clinical_data': total.carriers_with_clinical_data,
                'hy_available': total.hy_available,
                'hy_less_than_2': total.hy_less_than_2,
                'hy_less_than_3': total.hy_less_than_3,
                'moca_available': total.moca_available,
                'moca_gte_20': total.moca_gte_20,
                'moca_gte_24': total.moca_gte_24,
                'dat_caudate_available': total.dat_caudate_available,
                'disease_duration_lte_3_years': total.disease_duration_lte_3_years,
                'disease_duration_lte_5_years': total.disease_duration_lte_5_years,
                'disease_duration_lte_7_years': total.disease_duration_lte_7_years,
                'clinical_data_availability_pct': f"{total.clinical_data_availability_pct:.1f}%"
            }
            rows.append(row)

        return pd.DataFrame(rows)
