"""
Data models for NBA carriers data.
Defines dataclasses for the three combined NBA file types.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd


@dataclass
class NBAInfoRecord:
    """Data model for NBA info file records (variant metadata)."""
    
    # Core variant identifiers
    id: str
    snp_name: str
    snp_name_alt: Optional[str]
    locus: str
    rsid: Optional[str]
    
    # Genomic coordinates
    hg38: str
    hg19: str
    chrom: str
    pos: int
    a1: str  # reference allele
    a2: str  # alternate allele
    
    # Metadata
    ancestry: Optional[str]
    submitter_email: Optional[str]
    precision_medicine: Optional[str]
    pipeline: Optional[str]
    
    # Ancestry-specific frequency and quality metrics
    # Format: ALT_FREQS_{ANCESTRY}, OBS_CT_{ANCESTRY}, F_MISS_{ANCESTRY}, {ANCESTRY}_probe_used
    ancestry_metrics: Optional[Dict[str, Any]] = None
    
    # Overall metrics
    obs_ct: Optional[float] = None
    alt_freqs: Optional[float] = None
    f_miss: Optional[float] = None
    
    @classmethod
    def from_pandas_row(cls, row: pd.Series) -> 'NBAInfoRecord':
        """Create instance from pandas Series (DataFrame row)."""
        # Extract ancestry-specific metrics
        ancestry_metrics = {}
        ancestries = ['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN', 'MDE', 'SAS']
        
        for ancestry in ancestries:
            if f'ALT_FREQS_{ancestry}' in row.index:
                ancestry_metrics[ancestry] = {
                    'alt_freqs': row.get(f'ALT_FREQS_{ancestry}'),
                    'obs_ct': row.get(f'OBS_CT_{ancestry}'),
                    'f_miss': row.get(f'F_MISS_{ancestry}'),
                    'probe_used': row.get(f'{ancestry}_probe_used')
                }
        
        return cls(
            id=row['id'],
            snp_name=row['snp_name'],
            snp_name_alt=row.get('snp_name_alt'),
            locus=row['locus'],
            rsid=row.get('rsid') if pd.notna(row.get('rsid')) else None,
            hg38=row['hg38'],
            hg19=row['hg19'],
            chrom=row['chrom'],
            pos=int(row['pos']),
            a1=row['a1'],
            a2=row['a2'],
            ancestry=row.get('ancestry'),
            submitter_email=row.get('submitter_email'),
            precision_medicine=row.get('precision_medicine'),
            pipeline=row.get('pipeline'),
            ancestry_metrics=ancestry_metrics,
            obs_ct=row.get('OBS_CT'),
            alt_freqs=row.get('ALT_FREQS'),
            f_miss=row.get('F_MISS')
        )


@dataclass
class NBAIntRecord:
    """Data model for NBA int file records (integer genotype data)."""
    
    # Sample identifiers
    iid: str  # Individual ID
    ancestry: str
    
    # Genotype data (all variants as integers)
    # 0 = homozygous reference, 1 = heterozygous, 2 = homozygous alternate
    genotypes: Dict[str, Optional[int]]
    
    @classmethod
    def from_pandas_row(cls, row: pd.Series) -> 'NBAIntRecord':
        """Create instance from pandas Series (DataFrame row)."""
        # Extract genotype columns (all except IID and ancestry)
        genotype_cols = [col for col in row.index if col not in ['IID', 'ancestry']]
        genotypes = {}
        
        for col in genotype_cols:
            value = row[col]
            if pd.isna(value):
                genotypes[col] = None
            else:
                genotypes[col] = int(value) if value != '.' else None
        
        return cls(
            iid=row['IID'],
            ancestry=row['ancestry'],
            genotypes=genotypes
        )


@dataclass
class NBAStringRecord:
    """Data model for NBA string file records (string genotype data)."""
    
    # Sample identifiers  
    iid: str  # Individual ID
    ancestry: str
    
    # Genotype data (all variants as strings)
    # Typical values: "WT/WT", "WT/MUT", "MUT/MUT", "./."
    genotypes: Dict[str, Optional[str]]
    
    @classmethod
    def from_pandas_row(cls, row: pd.Series) -> 'NBAStringRecord':
        """Create instance from pandas Series (DataFrame row)."""
        # Extract genotype columns (all except IID and ancestry)
        genotype_cols = [col for col in row.index if col not in ['IID', 'ancestry']]
        genotypes = {}
        
        for col in genotype_cols:
            value = row[col]
            if pd.isna(value):
                genotypes[col] = None
            else:
                genotypes[col] = str(value) if value not in ['./.', '.', ''] else None
        
        return cls(
            iid=row['IID'],
            ancestry=row['ancestry'],
            genotypes=genotypes
        )


@dataclass
class NBADataset:
    """Container for all three NBA data types."""
    
    info_data: List[NBAInfoRecord]
    int_data: List[NBAIntRecord]
    string_data: List[NBAStringRecord]
    release: str
    
    @property
    def num_variants(self) -> int:
        """Number of variants in the dataset."""
        return len(self.info_data)
    
    @property
    def num_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.int_data)
    
    @property
    def variant_ids(self) -> List[str]:
        """List of all variant IDs."""
        return [record.id for record in self.info_data]
    
    @property
    def sample_ids(self) -> List[str]:
        """List of all sample IDs."""
        return [record.iid for record in self.int_data]
    
    @property
    def ancestries(self) -> List[str]:
        """List of unique ancestries in the dataset."""
        return list(set(record.ancestry for record in self.int_data))
    
    def get_variant_info(self, variant_id: str) -> Optional[NBAInfoRecord]:
        """Get variant info by ID."""
        for record in self.info_data:
            if record.id == variant_id:
                return record
        return None
    
    def get_sample_data(self, sample_id: str, data_type: str = 'int') -> Optional[dict]:
        """Get sample genotype data by ID.
        
        Args:
            sample_id: Sample identifier
            data_type: Either 'int' or 'string'
            
        Returns:
            Sample data record or None if not found
        """
        data_source = self.int_data if data_type == 'int' else self.string_data
        
        for record in data_source:
            if record.iid == sample_id:
                return record
        return None
