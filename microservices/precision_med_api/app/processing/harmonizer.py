import os
import pandas as pd
from typing import List, Dict, Any
import logging

from ..models.harmonization import (
    HarmonizationRecord, 
    HarmonizationAction
)
from ..core.config import Settings

logger = logging.getLogger(__name__)


class HarmonizationEngine:
    """Merge-based harmonization engine using direct allele comparison."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Complement map for strand flipping
        self.complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    
    def read_pvar_file(self, pgen_path: str) -> pd.DataFrame:
        """
        Read PVAR file for a PLINK file.
        
        Args:
            pgen_path: Path to .pgen file
            
        Returns:
            DataFrame with PVAR data
        """
        pvar_path = pgen_path.replace('.pgen', '.pvar')
        
        if not os.path.exists(pvar_path):
            raise FileNotFoundError(f"PVAR file not found: {pvar_path}")
        
        try:
            # Skip VCF-style comment lines (starting with ##) and find actual header/data
            skiprows = 0
            with open(pvar_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if line.startswith('##'):
                        skiprows += 1
                    else:
                        first_data_line = line.strip()
                        break
                        
            # Check if the first non-comment line contains headers (starts with #CHROM or has column names)
            has_headers = (first_data_line.startswith('#CHROM') or 
                          first_data_line.startswith('CHROM') or
                          any(col in first_data_line.upper() for col in ['CHROM', 'POS', 'ID', 'REF', 'ALT']))
            
            if has_headers:
                # Read with headers, skipping comment lines
                df = pd.read_csv(pvar_path, sep='\t', skiprows=skiprows, low_memory=False, dtype=str)
                # Handle column names (remove # prefix if present)
                df.columns = [col.lstrip('#') for col in df.columns]
            else:
                # Read without headers and assign standard column names, skipping comment lines  
                df = pd.read_csv(pvar_path, sep='\t', header=None, skiprows=skiprows, low_memory=False, dtype=str)
                # Assign standard PVAR column names
                if len(df.columns) >= 5:
                    df.columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT'] + [f'INFO_{i}' for i in range(len(df.columns)-5)]
                else:
                    raise ValueError(f"PVAR file has insufficient columns ({len(df.columns)}). Expected at least 5.")
            
            # Standardize column names
            expected_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT']
            if not all(col in df.columns for col in expected_cols):
                raise ValueError(f"PVAR file missing required columns: {expected_cols}. Found: {list(df.columns)}")
            
            # Clean and standardize data
            df['CHROM'] = df['CHROM'].astype(str).str.strip().str.replace('chr', '').str.upper()
            df['POS'] = pd.to_numeric(df['POS'], errors='coerce')
            df['ID'] = df['ID'].astype(str).str.strip()
            df['REF'] = df['REF'].astype(str).str.strip().str.upper()
            df['ALT'] = df['ALT'].astype(str).str.strip().str.upper()
            
            # Remove invalid rows
            df = df.dropna(subset=['CHROM', 'POS', 'REF', 'ALT'])
            
            logger.info(f"Read {len(df)} variants from {pvar_path}")

            return df
            
        except Exception as e:
            logger.error(f"Failed to read PVAR file {pvar_path}: {e}")
            raise
    
    def _prepare_snp_list(self, snp_list: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare SNP list DataFrame for merging.
        
        Args:
            snp_list: Raw SNP list DataFrame
            
        Returns:
            Cleaned SNP list DataFrame
        """
        df = snp_list.copy()
        
        # Standardize column names and data
        df['chromosome'] = df['chromosome'].astype(str).str.strip().str.replace('chr', '').str.upper()
        df['position'] = pd.to_numeric(df['position'], errors='coerce')
        df['ref'] = df['ref'].astype(str).str.strip().str.upper()
        df['alt'] = df['alt'].astype(str).str.strip().str.upper()
        
        # Remove invalid rows
        df = df.dropna(subset=['chromosome', 'position', 'ref', 'alt'])
        
        logger.info(f"Prepared {len(df)} SNP list variants for merging")
        
        return df
    
    def _merge_data(self, pvar_df: pd.DataFrame, snp_list: pd.DataFrame) -> pd.DataFrame:
        """
        Merge PVAR and SNP list data on chromosome and position.
        
        Args:
            pvar_df: PVAR DataFrame
            snp_list: SNP list DataFrame
            
        Returns:
            Merged DataFrame with both PVAR and SNP list information
        """
        # Prepare data for merge
        pvar_merge = pvar_df[['CHROM', 'POS', 'ID', 'REF', 'ALT']].copy()
        snp_merge = snp_list[['chromosome', 'position', 'variant_id', 'ref', 'alt']].copy()
        
        # Rename columns for consistent merge keys
        pvar_merge = pvar_merge.rename(columns={'CHROM': 'chromosome', 'POS': 'position'})
        snp_merge = snp_merge.rename(columns={'ref': 'a1', 'alt': 'a2'})
        
        # Merge on chromosome and position
        merged_df = pd.merge(
            pvar_merge, 
            snp_merge,
            on=['chromosome', 'position'],
            how='inner',
            suffixes=('_pvar', '_snp')
        )
        
        logger.info(f"Merged data: {len(merged_df)} matching variants found")
        return merged_df
    
    def _harmonize_on_merged(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Direct harmonization by comparing alleles in merged dataframe.
        Based on the user's provided function.
        """
        results = []
        
        for _, row in merged_df.iterrows():
            # PVAR alleles
            pvar_ref = row['REF'].upper()
            pvar_alt = row['ALT'].upper() 
            
            # SNP list alleles
            snp_ref = row['a1'].upper()
            snp_alt = row['a2'].upper()
            
            # Determine harmonization action
            if pvar_ref == snp_ref and pvar_alt == snp_alt:
                action = "EXACT"
                transform = None
            elif pvar_ref == snp_alt and pvar_alt == snp_ref:
                action = "SWAP"
                transform = "2-x"
            elif pvar_ref == self.complement_map.get(snp_ref, snp_ref) and pvar_alt == self.complement_map.get(snp_alt, snp_alt):
                action = "FLIP"
                transform = None
            elif pvar_ref == self.complement_map.get(snp_alt, snp_alt) and pvar_alt == self.complement_map.get(snp_ref, snp_ref):
                action = "FLIP_SWAP" 
                transform = "2-x"
            else:
                continue  # No valid transformation
                
            # Start with all columns from the merged row
            result_row = row.to_dict()
            
            # Add harmonization results
            result_row['harmonization_action'] = action
            result_row['genotype_transform'] = transform
            
            results.append(result_row)
        
        return pd.DataFrame(results)

    
    
    def harmonize_variants(
        self, 
        pvar_df: pd.DataFrame, 
        snp_list: pd.DataFrame
    ) -> List[HarmonizationRecord]:
        """
        Harmonize variants using merge-based approach.
        
        Args:
            pvar_df: DataFrame from PVAR file
            snp_list: Normalized SNP list DataFrame
            
        Returns:
            List of harmonization records for all valid matches
        """
        logger.info(f"Starting merge-based harmonization: {len(pvar_df)} PVAR variants vs {len(snp_list)} SNP list variants")
        
        # Step 1: Prepare SNP list data
        prepared_snp_list = self._prepare_snp_list(snp_list)
        
        # Step 2: Merge PVAR and SNP list data
        merged_df = self._merge_data(pvar_df, prepared_snp_list)
        
        if merged_df.empty:
            logger.warning("No variants found at matching positions")
            return []
        
        # Step 3: Perform harmonization on merged data
        harmonized_df = self._harmonize_on_merged(merged_df)
        
        if harmonized_df.empty:
            logger.warning("No variants could be harmonized")
            return []
        
        # Step 4: Convert to HarmonizationRecord objects
        records = []
        action_counts = {}
        
        for _, row in harmonized_df.iterrows():
            # Map string action to enum
            action_str = row['harmonization_action']
            if action_str == "EXACT":
                action_enum = HarmonizationAction.EXACT
            elif action_str == "SWAP":
                action_enum = HarmonizationAction.SWAP
            elif action_str == "FLIP":
                action_enum = HarmonizationAction.FLIP
            elif action_str == "FLIP_SWAP":
                action_enum = HarmonizationAction.FLIP_SWAP
            else:
                logger.warning(f"Unknown harmonization action: {action_str}")
                continue
            
            # Track action counts
            action_counts[action_str] = action_counts.get(action_str, 0) + 1
            
            # Create harmonization record
            record = HarmonizationRecord(
                snp_list_id=str(row['variant_id']),
                pgen_variant_id=str(row['ID']),
                chromosome=str(row['chromosome']),
                position=int(row['position']),
                snp_list_a1=str(row['a1']),
                snp_list_a2=str(row['a2']),
                pgen_a1=str(row['REF']),
                pgen_a2=str(row['ALT']),
                harmonization_action=action_enum,
                genotype_transform=row['genotype_transform'],
                file_path="",  # Set by caller
                data_type="",  # Set by caller  
                ancestry=None  # Set by caller
            )
            
            records.append(record)
        
        # Log summary
        logger.info(f"Merge-based harmonization complete: {len(records)} matches found")
        logger.info("Harmonization breakdown:")
        for action, count in action_counts.items():
            logger.info(f"  {action}: {count} variants")
        
        return records