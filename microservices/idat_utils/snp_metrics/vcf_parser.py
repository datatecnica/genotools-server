"""VCF Parser for SNP Metrics Extraction

Efficiently parses VCF files to extract key SNP metrics for downstream analysis.
Handles both gzipped and uncompressed VCF files.
"""

import gzip
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


class VCFParser:
    """Efficient VCF parser focused on SNP metrics extraction.
    
    Extracts essential columns for SNP analysis:
    - snpID: SNP identifier
    - chromosome: Chromosome number
    - position: Genomic position  
    - GT: Genotype (0/0, 0/1, 1/1, ./.)
    - GS: GenCall Score (confidence score for the genotype call)
    - BAF: B Allele Frequency
    - LRR: Log R Ratio
    """
    
    # Metrics we want to extract from the FORMAT field (based on actual VCF output)
    METRICS_COLUMNS = ['GT', 'GS', 'BAF', 'LRR']
    
    def parse_vcf(self, vcf_path: Union[str, Path], sample_id: Optional[str] = None) -> pd.DataFrame:
        """Parse VCF file and extract SNP metrics.
        
        Args:
            vcf_path: Path to VCF file (.vcf or .vcf.gz)
            sample_id: Sample identifier (extracted from filename if not provided)
            
        Returns:
            DataFrame with SNP metrics
            
        Raises:
            VCFParsingError: If VCF parsing fails
        """
        vcf_path = Path(vcf_path)
        
        if not vcf_path.exists():
            raise VCFParsingError(f"VCF file not found: {vcf_path}")
        
        if sample_id is None:
            sample_id = vcf_path.stem.replace('.vcf', '')
        
        try:
            # Read VCF data
            header, data_rows = self._read_vcf_data(vcf_path)
            
            # Create dataframe
            df = pd.DataFrame(data_rows, columns=header)
            
            # Extract and clean data
            df = self._extract_metrics(df, sample_id)
            
            return df
            
        except Exception as e:
            raise VCFParsingError(f"Failed to parse VCF {vcf_path}") from e
    
    def parse_variants(self, vcf_path: Union[str, Path]) -> pd.DataFrame:
        """Parse VCF file and extract variant-specific information only.
        
        Args:
            vcf_path: Path to VCF file (.vcf or .vcf.gz)
            
        Returns:
            DataFrame with variant information: #CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO
            
        Raises:
            VCFParsingError: If VCF parsing fails
        """
        vcf_path = Path(vcf_path)
        
        if not vcf_path.exists():
            raise VCFParsingError(f"VCF file not found: {vcf_path}")
        
        try:
            # Read VCF data
            header, data_rows = self._read_vcf_data(vcf_path)
            
            # Create dataframe
            df = pd.DataFrame(data_rows, columns=header)
            
            # Extract variant information only
            variant_df = self._extract_variant_info(df)
            
            return variant_df
            
        except Exception as e:
            raise VCFParsingError(f"Failed to parse VCF {vcf_path}") from e
    
    def _read_vcf_data(self, vcf_path: Path) -> tuple[List[str], List[List[str]]]:
        """Read VCF file and extract header and data rows."""
        opener = gzip.open if vcf_path.suffix == '.gz' else open
        mode = 'rt' if vcf_path.suffix == '.gz' else 'r'
        
        header = None
        data_rows = []
        
        with opener(vcf_path, mode) as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('#CHROM'):
                    # Found the column header line
                    header = line.split('\t')
                elif not line.startswith('#') and header is not None:
                    # Data line
                    data_rows.append(line.split('\t'))
        
        if header is None:
            raise VCFParsingError("No header line found in VCF file")
        
        return header, data_rows
    
    def _extract_variant_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract variant-specific information from VCF dataframe."""
        variant_cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
        
        # Extract variant information
        result_df = df[variant_cols].copy()
        
        # Clean chromosome names and convert position to numeric
        result_df['#CHROM'] = result_df['#CHROM'].str.replace('chr', '', regex=False)
        result_df['POS'] = pd.to_numeric(result_df['POS'], errors='coerce')
        
        # Rename #CHROM to chromosome for consistency
        result_df = result_df.rename(columns={'#CHROM': 'chromosome'})
        
        return result_df
    
    def _extract_metrics(self, df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
        """Extract and clean sample-specific metrics from VCF dataframe."""
        # Identify sample column (should be the last column after FORMAT)
        metadata_cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']
        sample_cols = [col for col in df.columns if col not in metadata_cols]
        
        if len(sample_cols) != 1:
            raise VCFParsingError(f"Expected 1 sample column, found {len(sample_cols)}: {sample_cols}")
        
        sample_col = sample_cols[0]
        
        # Extract sample-specific information only
        result_df = pd.DataFrame({
            'ID': df['ID'],
            'chromosome': df['#CHROM'].str.replace('chr', '', regex=False),
            'IID': sample_id
        })
        
        # Split the sample column by FORMAT specification
        format_fields = df['FORMAT'].iloc[0].split(':') if not df['FORMAT'].empty else []
        sample_data = df[sample_col].str.split(':', expand=True)
        
        # Map format fields to data columns
        for i, field in enumerate(format_fields):
            if field in self.METRICS_COLUMNS and i < sample_data.shape[1]:
                result_df[field] = sample_data[i]
        
        # Clean and convert data types
        result_df = self._clean_data_types(result_df)
        
        return result_df
    
    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data types for SNP metrics."""
        # Convert GT to numeric (0=0/0, 1=0/1, 2=1/1, -9=missing)
        if 'GT' in df.columns:
            gt_map = {'0/0': 0, '0/1': 1, '1/0': 1, '1/1': 2, './.': -9}
            df['GT'] = df['GT'].map(gt_map).fillna(-9).astype(int)
        
        # Convert numeric columns (GS, BAF, LRR are all numeric)
        numeric_columns = ['GS', 'BAF', 'LRR']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure chromosome is string
        df['chromosome'] = df['chromosome'].astype(str)
        
        return df


class VCFParsingError(Exception):
    """Exception raised for VCF parsing errors."""
    pass 