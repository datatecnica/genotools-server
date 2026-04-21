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
        self.output_formats = ['parquet']  # Only parquet format - contains all data
    
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
            'pgen_a1', 'pgen_a2', 'file_path', 'source_file',
            # TRAW format columns that are not sample genotypes
            '(C)M', 'COUNTED', 'ALT',
            # MAF correction columns
            'maf_corrected', 'original_alt_af'
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
                'variant_id': 'SNP',  # Use original PLINK variant ID from TRAW, not SNP list ID
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
            
            logger.debug(f"Wrote TRAW file with {len(traw_output)} variants and {len(sample_cols)} samples to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write TRAW file {output_path}: {e}")
            raise
    
    
    # write_harmonization_report method removed - was generating empty dictionaries
    
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
                
                # Create variant metadata summary (no population-specific statistics)
                summaries = []
                
                for _, row in df.iterrows():
                    summary = {
                        'variant_id': row.get('variant_id', '.'),
                        'snp_list_id': row.get('snp_list_id', '.'),
                        'chromosome': row.get('chromosome', '.'),
                        'position': row.get('position', '.'),
                        'original_a1': row.get('pgen_a1', '.'),
                        'original_a2': row.get('pgen_a2', '.'),
                        'counted_allele': row.get('counted_allele', '.'),
                        'alt_allele': row.get('alt_allele', '.'),
                        'harmonization_action': row.get('harmonization_action', '.'),
                        'data_type': row.get('data_type', '.'),
                        'source_file': row.get('source_file', '.')
                    }
                    summaries.append(summary)
                
                summary_df = pd.DataFrame(summaries)
            
            # Write summary
            if output_path.endswith('.parquet'):
                save_parquet(summary_df, output_path)
            else:
                summary_df.to_csv(output_path, index=False)
            
            logger.debug(f"Wrote variant summary with {len(summary_df)} variants to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write variant summary {output_path}: {e}")
            raise
    
    # _calculate_hwe_p method removed - population-specific statistics removed from variant summary
    
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
            
            # Harmonization report removed - contained only empty dictionaries
            
            # TFAM file generation removed - TRAW format is self-contained with sample IDs
                
        except Exception as e:
            logger.error(f"Failed to write additional files: {e}")
        
        logger.debug(f"Exported {len(output_files)} files to {output_dir}")
        return output_files
    
    # create_qc_report method removed - same information available in variant summary