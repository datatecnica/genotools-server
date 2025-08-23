"""
Output formatting for harmonized genomic data.

Formats extracted and harmonized variant data into standard formats
including PLINK TRAW, sample files, and harmonization reports.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

from ..models.harmonization import HarmonizationStats
from ..utils.parquet_io import save_parquet

logger = logging.getLogger(__name__)


class TrawFormatter:
    """Formats harmonized genotypes into PLINK TRAW format and reports."""
    
    def __init__(self):
        self.output_formats = ['traw', 'csv', 'parquet', 'json']
    
    def format_harmonized_genotypes(
        self, 
        df: pd.DataFrame, 
        snp_list: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Format harmonized genotypes for output.
        
        Args:
            df: DataFrame with harmonized genotypes
            snp_list: Original SNP list for metadata
            
        Returns:
            Formatted DataFrame ready for output
        """
        if df.empty:
            return df
        
        formatted_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['chromosome', 'variant_id', 'position']
        missing_cols = [col for col in required_cols if col not in formatted_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Standardize chromosome format
        formatted_df['chromosome'] = formatted_df['chromosome'].astype(str)
        
        # Add SNP list metadata if provided
        if snp_list is not None:
            snp_metadata = snp_list.set_index('variant_id')
            
            for col in ['gene', 'rsid', 'inheritance_pattern']:
                if col in snp_metadata.columns:
                    formatted_df[col] = formatted_df['snp_list_id'].map(
                        snp_metadata[col]
                    ).fillna('.')
        
        # Sort by chromosome and position
        formatted_df['chrom_sort'] = pd.Categorical(
            formatted_df['chromosome'],
            categories=[str(i) for i in range(1, 23)] + ['X', 'Y', 'MT'],
            ordered=True
        )
        formatted_df = formatted_df.sort_values(['chrom_sort', 'position'])
        formatted_df = formatted_df.drop(columns=['chrom_sort'])
        
        return formatted_df
    
    def _get_sample_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns containing sample genotypes."""
        metadata_cols = {
            'chromosome', 'variant_id', 'position', 'counted_allele', 'alt_allele',
            'harmonization_action', 'snp_list_id', 'data_type', 'ancestry',
            'chromosome_file', 'gene', 'rsid', 'inheritance_pattern',
            'genotype_transform', 'pgen_variant_id', 'snp_list_a1', 'snp_list_a2',
            'pgen_a1', 'pgen_a2', 'file_path',
            # TRAW format columns that are not sample genotypes
            '(C)M', 'COUNTED', 'ALT'
        }
        
        return [col for col in df.columns if col not in metadata_cols]
    
    def write_traw(
        self, 
        df: pd.DataFrame, 
        output_path: str,
        include_metadata: bool = True
    ) -> None:
        """
        Write DataFrame to PLINK TRAW format.
        
        TRAW format: CHR SNP (C)M POS COUNTED ALT [sample genotypes...]
        
        Args:
            df: Formatted genotype DataFrame
            output_path: Output file path
            include_metadata: Whether to include extra metadata columns
        """
        if df.empty:
            logger.warning("Empty DataFrame, creating empty TRAW file")
            Path(output_path).touch()
            return
        
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare TRAW DataFrame
            traw_df = df.copy()
            
            # Standard TRAW columns
            traw_columns = ['CHR', 'SNP', 'CM', 'POS', 'COUNTED', 'ALT']
            
            # Map our columns to TRAW format
            column_mapping = {
                'chromosome': 'CHR',
                'snp_list_id': 'SNP',  # Use original SNP list ID, not PLINK variant_id
                'position': 'POS',
                'counted_allele': 'COUNTED',
                'alt_allele': 'ALT'
            }
            
            # Build all columns efficiently using pd.concat
            column_dfs = []
            
            # Add mapped metadata columns
            metadata_dict = {}
            for our_col, traw_col in column_mapping.items():
                if our_col in traw_df.columns:
                    metadata_dict[traw_col] = traw_df[our_col]
                else:
                    metadata_dict[traw_col] = '.'
            
            # Add genetic distance (typically 0 for SNPs)
            metadata_dict['CM'] = 0
            
            # Create metadata DataFrame
            metadata_df = pd.DataFrame(metadata_dict)
            column_dfs.append(metadata_df)
            
            # Add sample genotype columns
            sample_cols = self._get_sample_columns(traw_df)
            if sample_cols:
                sample_df = traw_df[sample_cols].copy()
                column_dfs.append(sample_df)
            
            # Combine all columns at once using pd.concat
            if column_dfs:
                traw_output = pd.concat(column_dfs, axis=1)
            else:
                traw_output = metadata_df
            
            # Reorder columns
            final_columns = traw_columns + sample_cols
            traw_output = traw_output[final_columns]
            
            # Write to file
            traw_output.to_csv(output_path, sep='\t', index=False, na_rep='NA')
            
            logger.info(f"Wrote TRAW file with {len(traw_output)} variants and {len(sample_cols)} samples to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write TRAW file {output_path}: {e}")
            raise
    
    
    def write_harmonization_report(
        self, 
        harmonization_stats: Dict[str, Any], 
        output_path: str,
        detailed: bool = True
    ) -> None:
        """
        Write harmonization report with statistics and metadata.
        
        Args:
            harmonization_stats: Statistics from harmonization process
            output_path: Output file path
            detailed: Whether to include detailed breakdown
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create report structure
            report = {
                'generation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'generator': 'precision_med_api'
                },
                'summary': harmonization_stats.get('summary', {}),
                'harmonization_stats': harmonization_stats
            }
            
            if detailed:
                # Add detailed breakdown
                report['detailed_stats'] = {
                    'by_data_type': harmonization_stats.get('by_data_type', {}),
                    'by_ancestry': harmonization_stats.get('by_ancestry', {}),
                    'by_chromosome': harmonization_stats.get('by_chromosome', {}),
                    'transformation_summary': harmonization_stats.get('transformation_summary', {})
                }
            
            # Write JSON report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Wrote harmonization report to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write harmonization report {output_path}: {e}")
            raise
    
    def write_variant_summary(
        self, 
        df: pd.DataFrame, 
        output_path: str
    ) -> None:
        """
        Write variant-level summary statistics.
        
        Args:
            df: DataFrame with harmonized genotypes
            output_path: Output file path
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame, creating empty variant summary")
                summary_df = pd.DataFrame()
            else:
                # Get sample columns
                sample_cols = self._get_sample_columns(df)
                
                # Calculate summary statistics per variant
                summaries = []
                
                for _, row in df.iterrows():
                    # Get genotypes for this variant
                    genotypes = []
                    for col in sample_cols:
                        if pd.notna(row[col]):
                            genotypes.append(row[col])
                    
                    if genotypes:
                        # Convert to numeric, handling string values like "NA"
                        genotypes = pd.to_numeric(genotypes, errors='coerce')
                        genotypes = np.array(genotypes)
                        valid_gts = genotypes[~np.isnan(genotypes)]
                        
                        if len(valid_gts) > 0:
                            # Calculate statistics
                            n_samples = len(valid_gts)
                            n_missing = len(genotypes) - n_samples
                            
                            n_00 = np.sum(valid_gts == 0)
                            n_01 = np.sum(valid_gts == 1)
                            n_11 = np.sum(valid_gts == 2)
                            
                            alt_allele_count = 2 * n_11 + n_01
                            total_alleles = 2 * n_samples
                            alt_freq = alt_allele_count / total_alleles if total_alleles > 0 else 0
                            
                            summary = {
                                'variant_id': row.get('variant_id', '.'),
                                'snp_list_id': row.get('snp_list_id', '.'),
                                'chromosome': row.get('chromosome', '.'),
                                'position': row.get('position', '.'),
                                'counted_allele': row.get('counted_allele', '.'),
                                'alt_allele': row.get('alt_allele', '.'),
                                'harmonization_action': row.get('harmonization_action', '.'),
                                'data_type': row.get('data_type', '.'),
                                'ancestry': row.get('ancestry', '.'),
                                'n_samples': n_samples,
                                'n_missing': n_missing,
                                'missing_rate': n_missing / len(genotypes),
                                'n_00': n_00,
                                'n_01': n_01,
                                'n_11': n_11,
                                'alt_allele_count': alt_allele_count,
                                'alt_allele_freq': alt_freq,
                                'hwe_p': self._calculate_hwe_p(n_00, n_01, n_11)
                            }
                            summaries.append(summary)
                
                summary_df = pd.DataFrame(summaries)
            
            # Write summary
            if output_path.endswith('.parquet'):
                save_parquet(summary_df, output_path)
            else:
                summary_df.to_csv(output_path, index=False)
            
            logger.info(f"Wrote variant summary with {len(summary_df)} variants to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write variant summary {output_path}: {e}")
            raise
    
    def _calculate_hwe_p(self, n_00: int, n_01: int, n_11: int) -> float:
        """
        Calculate Hardy-Weinberg equilibrium p-value.
        
        Uses chi-square test for HWE.
        """
        try:
            total = n_00 + n_01 + n_11
            if total == 0:
                return 1.0
            
            # Calculate allele frequencies
            p = (2 * n_00 + n_01) / (2 * total)  # Frequency of allele 1
            q = 1 - p  # Frequency of allele 2
            
            # Expected counts under HWE
            exp_00 = total * p * p
            exp_01 = total * 2 * p * q
            exp_11 = total * q * q
            
            # Chi-square statistic
            chi_sq = 0
            if exp_00 > 0:
                chi_sq += (n_00 - exp_00) ** 2 / exp_00
            if exp_01 > 0:
                chi_sq += (n_01 - exp_01) ** 2 / exp_01
            if exp_11 > 0:
                chi_sq += (n_11 - exp_11) ** 2 / exp_11
            
            # Convert to p-value (1 degree of freedom)
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(chi_sq, df=1)
            
            return p_value
            
        except Exception:
            return 1.0  # Return 1.0 if calculation fails
    
    def export_multiple_formats(
        self, 
        df: pd.DataFrame, 
        output_dir: str,
        base_name: str,
        formats: List[str],
        snp_list: Optional[pd.DataFrame] = None,
        harmonization_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export data in multiple formats.
        
        Args:
            df: Harmonized genotype DataFrame
            output_dir: Output directory
            base_name: Base name for output files
            formats: List of formats to export
            snp_list: Original SNP list for metadata
            harmonization_stats: Harmonization statistics
            
        Returns:
            Dictionary mapping format to output file path
        """
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Format the data
        formatted_df = self.format_harmonized_genotypes(df, snp_list)
        
        output_files = {}
        
        for fmt in formats:
            try:
                if fmt == 'traw':
                    output_path = os.path.join(output_dir, f"{base_name}.traw")
                    self.write_traw(formatted_df, output_path)
                    output_files['traw'] = output_path
                    
                elif fmt == 'csv':
                    output_path = os.path.join(output_dir, f"{base_name}.csv")
                    formatted_df.to_csv(output_path, index=False)
                    output_files['csv'] = output_path
                    
                elif fmt == 'parquet':
                    output_path = os.path.join(output_dir, f"{base_name}.parquet")
                    save_parquet(formatted_df, output_path)
                    output_files['parquet'] = output_path
                    
                elif fmt == 'json':
                    output_path = os.path.join(output_dir, f"{base_name}.json")
                    formatted_df.to_json(output_path, orient='records', indent=2)
                    output_files['json'] = output_path
                    
                else:
                    logger.warning(f"Unknown output format: {fmt}")
                    
            except Exception as e:
                logger.error(f"Failed to export {fmt} format: {e}")
        
        # Write additional files
        try:
            # Variant summary
            summary_path = os.path.join(output_dir, f"{base_name}_variant_summary.csv")
            self.write_variant_summary(formatted_df, summary_path)
            output_files['variant_summary'] = summary_path
            
            # Harmonization report
            if harmonization_stats:
                report_path = os.path.join(output_dir, f"{base_name}_harmonization_report.json")
                self.write_harmonization_report(harmonization_stats, report_path)
                output_files['harmonization_report'] = report_path
            
            # TFAM file generation removed - TRAW format is self-contained with sample IDs
                
        except Exception as e:
            logger.error(f"Failed to write additional files: {e}")
        
        logger.info(f"Exported {len(output_files)} files to {output_dir}")
        return output_files
    
    def create_qc_report(
        self, 
        df: pd.DataFrame, 
        output_path: str
    ) -> None:
        """
        Create quality control report for extracted variants.
        
        Args:
            df: Harmonized genotype DataFrame
            output_path: Output file path
        """
        try:
            if df.empty:
                qc_stats = {'total_variants': 0}
            else:
                sample_cols = self._get_sample_columns(df)
                
                qc_stats = {
                    'total_variants': len(df),
                    'total_samples': len(sample_cols),
                    'data_types': df['data_type'].value_counts().to_dict() if 'data_type' in df.columns else {},
                    'ancestries': df['ancestry'].value_counts().to_dict() if 'ancestry' in df.columns else {},
                    'chromosomes': df['chromosome'].value_counts().to_dict() if 'chromosome' in df.columns else {},
                    'harmonization_actions': df['harmonization_action'].value_counts().to_dict() if 'harmonization_action' in df.columns else {}
                }
                
                # Calculate per-variant QC metrics
                if sample_cols:
                    missing_rates = []
                    alt_freqs = []
                    
                    for _, row in df.iterrows():
                        genotypes = [row[col] for col in sample_cols if pd.notna(row[col])]
                        if genotypes:
                            # Convert to numeric, handling string values like "NA"
                            genotypes = pd.to_numeric(genotypes, errors='coerce')
                            genotypes = np.array(genotypes)
                            valid_gts = genotypes[~np.isnan(genotypes)]
                            
                            missing_rate = 1 - (len(valid_gts) / len(genotypes))
                            missing_rates.append(missing_rate)
                            
                            if len(valid_gts) > 0:
                                alt_count = np.sum(valid_gts == 2) * 2 + np.sum(valid_gts == 1)
                                total_alleles = len(valid_gts) * 2
                                alt_freq = alt_count / total_alleles if total_alleles > 0 else 0
                                alt_freqs.append(alt_freq)
                    
                    if missing_rates:
                        qc_stats['missing_rate_stats'] = {
                            'mean': np.mean(missing_rates),
                            'median': np.median(missing_rates),
                            'max': np.max(missing_rates),
                            'variants_with_high_missing': np.sum(np.array(missing_rates) > 0.1)
                        }
                    
                    if alt_freqs:
                        qc_stats['allele_freq_stats'] = {
                            'mean': np.mean(alt_freqs),
                            'median': np.median(alt_freqs),
                            'min': np.min(alt_freqs),
                            'max': np.max(alt_freqs),
                            'rare_variants': np.sum(np.array(alt_freqs) < 0.01)
                        }
            
            # Add timestamp
            qc_stats['generation_timestamp'] = datetime.now().isoformat()
            
            # Write QC report
            with open(output_path, 'w') as f:
                json.dump(qc_stats, f, indent=2, default=str)
            
            logger.info(f"Wrote QC report to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write QC report {output_path}: {e}")
            raise