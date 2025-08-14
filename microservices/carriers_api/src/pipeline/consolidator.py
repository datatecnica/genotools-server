"""
Consolidation stage that merges PLINK files before processing.
"""

import tempfile
import shutil
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path

from .base import PipelineStage, PipelineContext
from ..utils import file_ops
from ..harmonization.harmonizer import HarmonizationService
from ..core.plink_operations import ExtractSnpsCommand, MergeCommand


class ConsolidationStage(PipelineStage[List[Tuple[str, Dict]], str]):
    """
    Stage that consolidates multiple PLINK files into a single file with target SNPs.
    """
    
    def __init__(self, harmonizer: HarmonizationService):
        super().__init__("Consolidation")
        self.harmonizer = harmonizer
    
    async def process(self, file_paths: List[Tuple[str, Dict]], context: PipelineContext) -> str:
        """
        Consolidate multiple PLINK files into single file with target SNPs.
        
        Args:
            file_paths: List of (path, metadata) tuples
            context: Pipeline context
            
        Returns:
            Path to consolidated PLINK file prefix
        """
        if len(file_paths) == 1:
            return await self._process_single_file(file_paths[0], context)
        
        return await self._consolidate_multiple_files(file_paths, context)
    
    async def _process_single_file(self, file_info: Tuple[str, Dict], context: PipelineContext) -> str:
        """Process single file by extracting target SNPs."""
        geno_path, metadata = file_info
        snplist_path = context.metadata.get('snplist_path')
        
        output_dir = Path(context.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        consolidated_prefix = str(output_dir / f"consolidated_{context.dataset_type}")
        
        # Extract target SNPs using harmonization
        subset_snp_path = await self.harmonizer.harmonize_and_extract(
            geno_path, snplist_path, consolidated_prefix
        )
        
        if subset_snp_path is None:
            raise ValueError(f"No target variants found in {geno_path}")
        
        return consolidated_prefix
    
    async def _consolidate_multiple_files(self, file_paths: List[Tuple[str, Dict]], 
                                        context: PipelineContext) -> str:
        """Consolidate multiple files."""
        tmpdir = tempfile.mkdtemp()
        snplist_path = context.metadata.get('snplist_path')
        
        try:
            # Extract target SNPs from each file
            extracted_files = []
            
            for geno_path, metadata in file_paths:
                # Create unique prefix for this file
                file_id = self._get_file_identifier(metadata)
                extracted_prefix = str(Path(tmpdir) / f"extracted_{file_id}")
                
                # Extract SNPs
                subset_snp_path = await self.harmonizer.harmonize_and_extract(
                    geno_path, snplist_path, extracted_prefix
                )
                
                if subset_snp_path:
                    extracted_files.append(extracted_prefix)
                else:
                    self.logger.warning(f"No variants found in {geno_path}")
            
            if not extracted_files:
                raise ValueError("No variants found in any input files")
            
            # Merge extracted files
            output_dir = Path(context.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            consolidated_prefix = str(output_dir / f"consolidated_{context.dataset_type}")
            
            if len(extracted_files) == 1:
                # Just copy single file
                for ext in ['.pgen', '.pvar', '.psam']:
                    shutil.copy2(f"{extracted_files[0]}{ext}", f"{consolidated_prefix}{ext}")
            else:
                # Merge multiple files
                merge_cmd = MergeCommand(extracted_files, consolidated_prefix)
                merge_cmd.execute()
            
            return consolidated_prefix
            
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    def _get_file_identifier(self, metadata: Dict) -> str:
        """Generate unique identifier for a file based on metadata."""
        parts = []
        if 'ancestry' in metadata:
            parts.append(metadata['ancestry'])
        if 'chromosome' in metadata:
            parts.append(f"chr{metadata['chromosome']}")
        return "_".join(parts) if parts else "file"
