"""
Variant harmonization cache building and management.

Builds caches that map variants between SNP lists and PLINK files,
handling allele orientation differences due to strand flips and allele swaps.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import subprocess
import tempfile

from ..models.harmonization import (
    HarmonizationRecord, 
    HarmonizationAction, 
    HarmonizationStats
)
from ..models.analysis import DataType
from ..core.config import Settings
from ..utils.parquet_io import save_parquet, read_parquet, optimize_dtypes_for_genomics

logger = logging.getLogger(__name__)


class AlleleHarmonizer:
    """Core allele harmonization logic."""
    
    # Base complements for strand flipping
    COMPLEMENT_MAP = {
        'A': 'T', 'T': 'A',
        'C': 'G', 'G': 'C',
        'N': 'N', '.': '.'
    }
    
    # Strand ambiguous allele pairs
    AMBIGUOUS_PAIRS = {('A', 'T'), ('T', 'A'), ('C', 'G'), ('G', 'C')}
    
    @classmethod
    def complement_allele(cls, allele: str) -> str:
        """Get complement of an allele for strand flipping."""
        return cls.COMPLEMENT_MAP.get(allele.upper(), allele)
    
    @classmethod
    def check_strand_ambiguous(cls, a1: str, a2: str) -> bool:
        """Check if allele pair is strand ambiguous (A/T or C/G)."""
        pair = (a1.upper(), a2.upper())
        return pair in cls.AMBIGUOUS_PAIRS or (pair[1], pair[0]) in cls.AMBIGUOUS_PAIRS
    
    @classmethod
    def get_all_representations(cls, a1: str, a2: str) -> List[Tuple[str, str, str]]:
        """
        Get all possible representations of an allele pair.
        
        Returns:
            List of (a1, a2, action) tuples for all possible orientations
        """
        a1, a2 = a1.upper(), a2.upper()
        
        representations = [
            (a1, a2, "EXACT"),                                      # Original
            (a2, a1, "SWAP"),                                       # Swapped
            (cls.complement_allele(a1), cls.complement_allele(a2), "FLIP"),     # Flipped
            (cls.complement_allele(a2), cls.complement_allele(a1), "FLIP_SWAP") # Flipped + Swapped
        ]
        
        return representations
    
    @classmethod
    def determine_harmonization(
        cls, 
        snp_a1: str, 
        snp_a2: str, 
        pgen_a1: str, 
        pgen_a2: str
    ) -> Tuple[str, Optional[str]]:
        """
        Determine harmonization action needed.
        
        Args:
            snp_a1, snp_a2: Alleles from SNP list (reference orientation)
            pgen_a1, pgen_a2: Alleles from PLINK file
            
        Returns:
            (action, genotype_transform) tuple
        """
        snp_a1, snp_a2 = snp_a1.upper().strip(), snp_a2.upper().strip()
        pgen_a1, pgen_a2 = pgen_a1.upper().strip(), pgen_a2.upper().strip()
        
        # Check for strand ambiguity
        if cls.check_strand_ambiguous(snp_a1, snp_a2):
            logger.warning(f"Strand ambiguous variant: {snp_a1}/{snp_a2}")
            return "AMBIGUOUS", None
        
        # Get all possible representations of the SNP list variant
        representations = cls.get_all_representations(snp_a1, snp_a2)
        
        # Try to match PLINK alleles to any representation
        for rep_a1, rep_a2, action in representations:
            if (rep_a1 == pgen_a1 and rep_a2 == pgen_a2):
                # Determine genotype transformation
                transform = None
                if action in ("SWAP", "FLIP_SWAP"):
                    transform = "2-x"  # 0->2, 1->1, 2->0
                
                return action, transform
        
        # No match found
        logger.warning(f"Cannot harmonize {snp_a1}/{snp_a2} with {pgen_a1}/{pgen_a2}")
        return "INVALID", None


class CacheBuilder:
    """Builds and manages variant harmonization caches."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.harmonizer = AlleleHarmonizer()
    
    def _read_pvar_file(self, pvar_path: str) -> pd.DataFrame:
        """Read PVAR file and normalize format."""
        try:
            # Read PVAR file (tab-separated, may have header starting with #)
            with open(pvar_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines starting with #
            data_lines = [line for line in lines if not line.startswith('#')]
            
            # Create DataFrame
            columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT']
            if len(data_lines) > 0:
                first_line = data_lines[0].strip().split('\t')
                if len(first_line) > 5:
                    columns.extend([f'COL{i}' for i in range(6, len(first_line))])
            
            # Parse data
            data = []
            for line in data_lines:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    data.append(parts[:len(columns)])
            
            df = pd.DataFrame(data, columns=columns[:len(data[0]) if data else 5])
            
            # Normalize data types
            df['CHROM'] = df['CHROM'].str.replace('chr', '').str.upper()
            df['POS'] = pd.to_numeric(df['POS'], errors='coerce')
            df['REF'] = df['REF'].str.upper().str.strip()
            df['ALT'] = df['ALT'].str.upper().str.strip()
            
            # Filter out invalid rows
            df = df.dropna(subset=['CHROM', 'POS', 'REF', 'ALT'])
            
            logger.info(f"Read {len(df)} variants from {pvar_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read PVAR file {pvar_path}: {e}")
            raise
    
    def _normalize_snp_list(self, snp_list: pd.DataFrame) -> pd.DataFrame:
        """Normalize SNP list format for matching."""
        df = snp_list.copy()
        
        # Extract coordinates from hg38 if needed
        if 'hg38' in df.columns and 'chromosome' not in df.columns:
            coords = df['hg38'].str.split(':', expand=True)
            df['chromosome'] = coords[0].str.replace('chr', '').str.upper()
            df['position'] = pd.to_numeric(coords[1], errors='coerce')
            df['ref'] = coords[2].str.upper().str.strip()
            df['alt'] = coords[3].str.upper().str.strip()
        
        # Ensure required columns exist
        required_cols = ['chromosome', 'position', 'ref', 'alt']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"SNP list missing required column: {col}")
        
        # Create variant ID if not present
        if 'variant_id' not in df.columns:
            df['variant_id'] = (
                df['chromosome'].astype(str) + ':' +
                df['position'].astype(str) + ':' +
                df['ref'].astype(str) + ':' +
                df['alt'].astype(str)
            )
        
        # Normalize data types
        df['chromosome'] = df['chromosome'].astype(str).str.replace('chr', '').str.upper()
        df['position'] = pd.to_numeric(df['position'], errors='coerce')
        df['ref'] = df['ref'].astype(str).str.upper().str.strip()
        df['alt'] = df['alt'].astype(str).str.upper().str.strip()
        
        # Remove invalid rows
        df = df.dropna(subset=['chromosome', 'position', 'ref', 'alt'])
        
        return df
    
    def match_variants_with_harmonization(
        self, 
        pvar_df: pd.DataFrame, 
        snp_list: pd.DataFrame
    ) -> List[HarmonizationRecord]:
        """
        Match PVAR variants to SNP list with harmonization.
        
        Args:
            pvar_df: DataFrame from PVAR file
            snp_list: Normalized SNP list DataFrame
            
        Returns:
            List of harmonization records
        """
        records = []
        
        # Create lookup for SNP list variants by position
        snp_lookup = {}
        for _, snp in snp_list.iterrows():
            key = f"{snp['chromosome']}:{snp['position']}"
            if key not in snp_lookup:
                snp_lookup[key] = []
            snp_lookup[key].append(snp)
        
        # Group PVAR variants by position to handle multi-allelic sites
        pvar_by_position = {}
        for _, pvar_row in pvar_df.iterrows():
            pos_key = f"{pvar_row['CHROM']}:{pvar_row['POS']}"
            if pos_key not in pvar_by_position:
                pvar_by_position[pos_key] = []
            pvar_by_position[pos_key].append(pvar_row)
        
        logger.info(f"Matching {len(pvar_df)} PVAR variants against {len(snp_list)} SNP list variants")
        
        # Match variants with improved multi-allelic handling
        matched_count = 0
        multi_allelic_count = 0
        
        for pos_key, pvar_variants in pvar_by_position.items():
            if len(pvar_variants) > 1:
                multi_allelic_count += 1
                logger.debug(f"Multi-allelic position {pos_key}: {len(pvar_variants)} PLINK variants")
            
            if pos_key in snp_lookup:
                # Try to match each SNP at this position with each PLINK variant
                for snp_row in snp_lookup[pos_key]:
                    best_match = None
                    best_score = 0
                    
                    for pvar_row in pvar_variants:
                        action, transform = self.harmonizer.determine_harmonization(
                            snp_row['ref'], snp_row['alt'],
                            pvar_row['REF'], pvar_row['ALT']
                        )
                        
                        if action != "INVALID":
                            # Calculate quality score for this match
                            score = self._calculate_match_quality(action, pvar_row)
                            
                            if score > best_score:
                                best_score = score
                                best_match = {
                                    'pvar_row': pvar_row,
                                    'action': action,
                                    'transform': transform
                                }
                    
                    # Create record for best match if found
                    if best_match:
                        record = HarmonizationRecord(
                            snp_list_id=snp_row['variant_id'],
                            pgen_variant_id=best_match['pvar_row']['ID'],
                            chromosome=str(best_match['pvar_row']['CHROM']),
                            position=int(best_match['pvar_row']['POS']),
                            snp_list_a1=snp_row['ref'],
                            snp_list_a2=snp_row['alt'],
                            pgen_a1=best_match['pvar_row']['REF'],
                            pgen_a2=best_match['pvar_row']['ALT'],
                            harmonization_action=HarmonizationAction(best_match['action']),
                            genotype_transform=best_match['transform'],
                            file_path="",  # Will be set by caller
                            data_type="",  # Will be set by caller
                            ancestry=None  # Will be set by caller
                        )
                        records.append(record)
                        matched_count += 1
        
        logger.info(f"Matched {matched_count} variants with harmonization")
        logger.info(f"Processed {multi_allelic_count} multi-allelic positions")
        return records

    def _calculate_match_quality(self, action: str, pvar_row: pd.Series) -> float:
        """
        Calculate quality score for a harmonization match.
        Higher scores are better.
        
        Args:
            action: Harmonization action (EXACT, FLIP, SWAP, etc.)
            pvar_row: PVAR row data
            
        Returns:
            Quality score (0-100)
        """
        base_scores = {
            'EXACT': 100,
            'FLIP': 90,
            'SWAP': 80,
            'FLIP_SWAP': 70,
            'AMBIGUOUS': 20,
            'INVALID': 0
        }
        
        score = base_scores.get(action, 0)
        
        # Prefer rsIDs over other variant IDs
        variant_id = str(pvar_row['ID'])
        if variant_id.startswith('rs'):
            score += 10
        elif 'ilmn' in variant_id.lower():
            score += 5
        
        # Prefer non-symbolic alleles
        if pvar_row['REF'] not in ['D', 'I', 'N'] and pvar_row['ALT'] not in ['D', 'I', 'N']:
            score += 5
        
        return score
    
    def build_harmonization_cache(
        self, 
        pvar_path: str, 
        snp_list: pd.DataFrame,
        data_type: str,
        ancestry: Optional[str] = None
    ) -> Tuple[pd.DataFrame, HarmonizationStats]:
        """
        Build harmonization cache for a single PLINK file.
        
        Args:
            pvar_path: Path to PVAR file
            snp_list: SNP list DataFrame
            data_type: Data type (NBA, WGS, IMPUTED)
            ancestry: Ancestry (for NBA/IMPUTED)
            
        Returns:
            (cache_df, stats) tuple
        """
        start_time = time.time()
        
        logger.info(f"Building harmonization cache for {pvar_path}")
        
        # Read and normalize inputs
        pvar_df = self._read_pvar_file(pvar_path)
        snp_list_norm = self._normalize_snp_list(snp_list)
        
        # Match variants with harmonization
        records = self.match_variants_with_harmonization(pvar_df, snp_list_norm)
        
        # Set file metadata
        base_path = os.path.dirname(pvar_path)
        pgen_path = pvar_path.replace('.pvar', '.pgen')
        
        for record in records:
            record.file_path = pgen_path
            record.data_type = data_type
            record.ancestry = ancestry
        
        # Convert to DataFrame
        if records:
            cache_df = pd.DataFrame([record.model_dump() for record in records])
            cache_df = optimize_dtypes_for_genomics(cache_df)
        else:
            # Empty DataFrame with correct schema
            cache_df = pd.DataFrame(columns=[
                'snp_list_id', 'pgen_variant_id', 'chromosome', 'position',
                'snp_list_a1', 'snp_list_a2', 'pgen_a1', 'pgen_a2',
                'harmonization_action', 'genotype_transform', 'file_path',
                'data_type', 'ancestry', 'harmonized_file_path'
            ])
        
        # Generate harmonized PLINK files if we have harmonization records
        harmonized_pgen_path = None
        if records and self._check_plink2_availability():
            try:
                logger.info("Generating harmonized PLINK files...")
                harmonized_pgen_path = self.generate_harmonized_plink_files(pgen_path, cache_df)
                if harmonized_pgen_path:
                    # Update cache with harmonized file path
                    cache_df['harmonized_file_path'] = harmonized_pgen_path
                    logger.info(f"Generated harmonized PLINK files: {harmonized_pgen_path}")
                else:
                    logger.warning("Failed to generate harmonized PLINK files, will use raw extraction")
                    cache_df['harmonized_file_path'] = None
            except Exception as e:
                logger.error(f"Error generating harmonized PLINK files: {e}")
                cache_df['harmonized_file_path'] = None
        else:
            cache_df['harmonized_file_path'] = None
        
        # Generate statistics
        stats = HarmonizationStats(total_variants=len(snp_list_norm))
        if records:
            stats.update_from_records(records)
        stats.processing_time_seconds = time.time() - start_time
        if harmonized_pgen_path:
            stats.harmonized_files_generated = True
        
        logger.info(f"Built cache with {len(records)} harmonized variants in {stats.processing_time_seconds:.1f}s")
        
        return cache_df, stats
    
    def save_cache(
        self, 
        cache_df: pd.DataFrame, 
        cache_path: str,
        stats: Optional[HarmonizationStats] = None
    ) -> None:
        """Save harmonization cache to parquet file."""
        try:
            # Ensure directory exists
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save cache
            save_parquet(cache_df, cache_path, compression="snappy")
            
            # Save stats if provided
            if stats:
                stats.cache_file_path = cache_path
                stats_path = cache_path.replace('.parquet', '_stats.json')
                with open(stats_path, 'w') as f:
                    f.write(stats.model_dump_json(indent=2))
            
            logger.info(f"Saved harmonization cache to {cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}")
            raise
    
    def load_cache(self, cache_path: str) -> pd.DataFrame:
        """Load harmonization cache from parquet file."""
        try:
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Cache file not found: {cache_path}")
            
            cache_df = read_parquet(cache_path)
            logger.info(f"Loaded {len(cache_df)} harmonization records from {cache_path}")
            return cache_df
            
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_path}: {e}")
            raise
    
    def validate_harmonization(self, cache_df: pd.DataFrame) -> Dict[str, int]:
        """Validate harmonization cache and return statistics."""
        if cache_df.empty:
            return {"total": 0}
        
        stats = {}
        stats["total"] = len(cache_df)
        stats["unique_snp_list_variants"] = cache_df['snp_list_id'].nunique()
        stats["unique_pgen_variants"] = cache_df['pgen_variant_id'].nunique()
        
        # Count by harmonization action
        action_counts = cache_df['harmonization_action'].value_counts().to_dict()
        for action in HarmonizationAction:
            stats[f"action_{action.value.lower()}"] = action_counts.get(action.value, 0)
        
        # Count variants requiring transformation
        stats["requires_transformation"] = len(
            cache_df[cache_df['genotype_transform'].notna()]
        )
        
        # Count by chromosome
        chrom_counts = cache_df['chromosome'].value_counts().to_dict()
        stats["chromosomes"] = chrom_counts
        
        return stats
    
    def _check_plink2_availability(self) -> bool:
        """Check if PLINK 2.0 is available."""
        try:
            result = subprocess.run(['plink2', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("PLINK 2.0 not found, harmonized PLINK files will not be generated")
            return False
    
    def _get_harmonized_plink_path(self, original_pgen_path: str) -> str:
        """Get path for harmonized PLINK files."""
        # Place harmonized files in same directory with _harmonized suffix
        base_path = original_pgen_path.replace('.pgen', '_harmonized')
        return base_path
    
    def _create_flip_lists(self, cache_df: pd.DataFrame, temp_dir: str) -> Dict[str, str]:
        """
        Create flip lists for different harmonization actions.
        
        Args:
            cache_df: Harmonization cache DataFrame
            temp_dir: Temporary directory for flip list files
            
        Returns:
            Dictionary mapping action to flip list file path
        """
        flip_files = {}
        
        # Group variants by harmonization action
        action_groups = cache_df.groupby('harmonization_action')
        
        for action, group in action_groups:
            if action in ['FLIP', 'FLIP_SWAP']:
                # Create flip list file
                flip_file = os.path.join(temp_dir, f"flip_{action.lower()}.txt")
                variant_ids = group['pgen_variant_id'].tolist()
                
                with open(flip_file, 'w') as f:
                    for var_id in variant_ids:
                        f.write(f"{var_id}\n")
                
                flip_files[action] = flip_file
                logger.debug(f"Created flip list for {action}: {len(variant_ids)} variants")
        
        return flip_files
    
    def _create_allele_lists(self, cache_df: pd.DataFrame, temp_dir: str) -> Dict[str, str]:
        """
        Create allele reference lists for harmonization.
        
        Args:
            cache_df: Harmonization cache DataFrame
            temp_dir: Temporary directory for allele list files
            
        Returns:
            Dictionary mapping action to allele list file path
        """
        allele_files = {}
        
        # Create reference allele list for consistent orientation
        ref_allele_file = os.path.join(temp_dir, "ref_alleles.txt")
        
        with open(ref_allele_file, 'w') as f:
            for _, row in cache_df.iterrows():
                # Write variant with reference allele from SNP list
                f.write(f"{row['pgen_variant_id']}\t{row['snp_list_a1']}\n")
        
        allele_files['reference'] = ref_allele_file
        logger.debug(f"Created reference allele list: {len(cache_df)} variants")
        
        return allele_files
    
    def generate_harmonized_plink_files(
        self, 
        original_pgen_path: str, 
        cache_df: pd.DataFrame
    ) -> Optional[str]:
        """
        Generate harmonized PLINK files using PLINK2 commands.
        
        Args:
            original_pgen_path: Path to original PGEN file
            cache_df: Harmonization cache DataFrame
            
        Returns:
            Path to harmonized PGEN file or None if failed
        """
        if not self._check_plink2_availability():
            logger.warning("PLINK2 not available, skipping harmonized PLINK generation")
            return None
        
        if cache_df.empty:
            logger.warning("Empty harmonization cache, no harmonized PLINK files to generate")
            return None
        
        try:
            # Get paths
            original_base = original_pgen_path.replace('.pgen', '')
            harmonized_base = self._get_harmonized_plink_path(original_pgen_path)
            
            # Ensure output directory exists
            Path(harmonized_base).parent.mkdir(parents=True, exist_ok=True)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create flip and allele lists
                flip_files = self._create_flip_lists(cache_df, temp_dir)
                allele_files = self._create_allele_lists(cache_df, temp_dir)
                
                # Start with original files
                current_base = original_base
                temp_base = os.path.join(temp_dir, "temp_harmonized")
                
                # Apply transformations step by step
                
                # Step 1: Apply strand flips if needed
                if 'FLIP' in flip_files or 'FLIP_SWAP' in flip_files:
                    # Combine all flip variants
                    all_flip_file = os.path.join(temp_dir, "all_flips.txt")
                    with open(all_flip_file, 'w') as outf:
                        for action in ['FLIP', 'FLIP_SWAP']:
                            if action in flip_files:
                                with open(flip_files[action], 'r') as inf:
                                    outf.write(inf.read())
                    
                    # Apply flips
                    cmd_flip = [
                        'plink2',
                        '--pfile', current_base,
                        '--flip', all_flip_file,
                        '--make-pgen',
                        '--out', f"{temp_base}_flipped"
                    ]
                    
                    result = subprocess.run(cmd_flip, capture_output=True, text=True, timeout=600)
                    if result.returncode != 0:
                        logger.error(f"PLINK2 flip command failed: {result.stderr}")
                        return None
                    
                    current_base = f"{temp_base}_flipped"
                    logger.info("Applied strand flips to harmonized PLINK files")
                
                # Step 2: Set reference alleles for consistent orientation
                if 'reference' in allele_files:
                    cmd_ref = [
                        'plink2',
                        '--pfile', current_base,
                        '--ref-allele', allele_files['reference'],
                        '--make-pgen',
                        '--out', harmonized_base
                    ]
                    
                    result = subprocess.run(cmd_ref, capture_output=True, text=True, timeout=600)
                    if result.returncode != 0:
                        logger.error(f"PLINK2 ref-allele command failed: {result.stderr}")
                        return None
                    
                    logger.info("Applied reference allele orientation to harmonized PLINK files")
                else:
                    # Just copy files if no reference alleles needed
                    cmd_copy = [
                        'plink2',
                        '--pfile', current_base,
                        '--make-pgen',
                        '--out', harmonized_base
                    ]
                    
                    result = subprocess.run(cmd_copy, capture_output=True, text=True, timeout=600)
                    if result.returncode != 0:
                        logger.error(f"PLINK2 copy command failed: {result.stderr}")
                        return None
                
                # Verify output files exist
                harmonized_pgen = f"{harmonized_base}.pgen"
                harmonized_pvar = f"{harmonized_base}.pvar"
                harmonized_psam = f"{harmonized_base}.psam"
                
                if all(os.path.exists(f) for f in [harmonized_pgen, harmonized_pvar, harmonized_psam]):
                    logger.info(f"Successfully generated harmonized PLINK files: {harmonized_base}")
                    return harmonized_pgen
                else:
                    logger.error("Harmonized PLINK files were not created properly")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to generate harmonized PLINK files: {e}")
            return None
    
    def _get_cache_path(self, data_type: str, release: str, ancestry: Optional[str] = None, chrom: Optional[str] = None) -> str:
        """Generate cache file path."""
        cache_base = os.path.join(self.settings.get_cache_path(), f"release{release}")
        
        if data_type == "WGS":
            return os.path.join(cache_base, "wgs", "wgs_variant_harmonization.parquet")
        elif data_type == "NBA":
            if not ancestry:
                raise ValueError("Ancestry required for NBA cache path")
            return os.path.join(cache_base, "nba", f"{ancestry}_variant_harmonization.parquet")
        elif data_type == "IMPUTED":
            if not ancestry or not chrom:
                raise ValueError("Ancestry and chromosome required for IMPUTED cache path")
            return os.path.join(cache_base, "imputed", ancestry, f"chr{chrom}_variant_harmonization.parquet")
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    
    def _get_file_paths_for_data_type(self, data_type: str, ancestry: Optional[str] = None, chrom: Optional[str] = None) -> List[str]:
        """Get PLINK file paths for a data type."""
        if data_type == "WGS":
            base_path = self.settings.get_wgs_path()
            return [f"{base_path}.pvar"]
        elif data_type == "NBA":
            if not ancestry:
                return []
            base_path = self.settings.get_nba_path(ancestry)
            return [f"{base_path}.pvar"]
        elif data_type == "IMPUTED":
            if not ancestry or not chrom:
                return []
            base_path = self.settings.get_imputed_path(ancestry, chrom)
            return [f"{base_path}.pvar"]
        else:
            return []
    
    def build_all_harmonization_caches(
        self, 
        snp_list: pd.DataFrame, 
        data_type: DataType, 
        release: str,
        force_rebuild: bool = False,
        max_workers: int = 4
    ) -> Dict[str, HarmonizationStats]:
        """
        Build harmonization caches for all files of a data type.
        
        Args:
            snp_list: SNP list DataFrame
            data_type: Data type to process
            release: Release version
            force_rebuild: Force rebuild even if cache exists
            max_workers: Maximum parallel workers
            
        Returns:
            Dictionary of file_path -> HarmonizationStats
        """
        all_stats = {}
        
        if data_type == DataType.WGS:
            # Single WGS file
            cache_path = self._get_cache_path("WGS", release)
            pvar_paths = self._get_file_paths_for_data_type("WGS")
            
            if pvar_paths and (force_rebuild or not os.path.exists(cache_path)):
                cache_df, stats = self.build_harmonization_cache(
                    pvar_paths[0], snp_list, "WGS"
                )
                self.save_cache(cache_df, cache_path, stats)
                all_stats[pvar_paths[0]] = stats
        
        elif data_type == DataType.NBA:
            # NBA files by ancestry
            tasks = []
            for ancestry in self.settings.list_available_ancestries("NBA"):
                cache_path = self._get_cache_path("NBA", release, ancestry=ancestry)
                pvar_paths = self._get_file_paths_for_data_type("NBA", ancestry=ancestry)
                
                if pvar_paths and (force_rebuild or not os.path.exists(cache_path)):
                    tasks.append((pvar_paths[0], ancestry, cache_path))
            
            # Process in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(
                        self.build_harmonization_cache, 
                        pvar_path, snp_list, "NBA", ancestry
                    ): (pvar_path, cache_path)
                    for pvar_path, ancestry, cache_path in tasks
                }
                
                for future in as_completed(future_to_task):
                    pvar_path, cache_path = future_to_task[future]
                    try:
                        cache_df, stats = future.result()
                        self.save_cache(cache_df, cache_path, stats)
                        all_stats[pvar_path] = stats
                    except Exception as e:
                        logger.error(f"Failed to build cache for {pvar_path}: {e}")
        
        elif data_type == DataType.IMPUTED:
            # IMPUTED files by ancestry and chromosome
            tasks = []
            for ancestry in self.settings.list_available_ancestries("IMPUTED"):
                for chrom in self.settings.list_available_chromosomes(ancestry):
                    cache_path = self._get_cache_path("IMPUTED", release, ancestry=ancestry, chrom=chrom)
                    pvar_paths = self._get_file_paths_for_data_type("IMPUTED", ancestry=ancestry, chrom=chrom)
                    
                    if pvar_paths and (force_rebuild or not os.path.exists(cache_path)):
                        tasks.append((pvar_paths[0], ancestry, cache_path))
            
            # Process in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(
                        self.build_harmonization_cache,
                        pvar_path, snp_list, "IMPUTED", ancestry
                    ): (pvar_path, cache_path)
                    for pvar_path, ancestry, cache_path in tasks
                }
                
                for future in as_completed(future_to_task):
                    pvar_path, cache_path = future_to_task[future]
                    try:
                        cache_df, stats = future.result()
                        self.save_cache(cache_df, cache_path, stats)
                        all_stats[pvar_path] = stats
                    except Exception as e:
                        logger.error(f"Failed to build cache for {pvar_path}: {e}")
        
        logger.info(f"Built {len(all_stats)} harmonization caches for {data_type.value}")
        return all_stats