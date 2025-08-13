"""
Variant data models.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd


@dataclass
class Variant:
    """Represents a single genetic variant."""
    id: str  # Variant ID used in genotype files
    chrom: str
    pos: int
    ref: str
    alt: str
    rsid: Optional[str] = None
    hg38: Optional[str] = None
    hg19: Optional[str] = None
    locus: Optional[str] = None
    snp_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'chrom': self.chrom,
            'pos': self.pos,
            'ref': self.ref,
            'alt': self.alt,
            'rsid': self.rsid,
            'hg38': self.hg38,
            'hg19': self.hg19,
            'locus': self.locus,
            'snp_name': self.snp_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Variant':
        """Create from dictionary representation."""
        return cls(**data)
    
    def get_hg38_format(self) -> str:
        """Get variant in hg38 format (chr:pos:ref:alt)."""
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"


@dataclass
class VariantStats:
    """Statistics for a variant."""
    variant_id: str
    alt_freq: Optional[float] = None
    obs_ct: Optional[int] = None
    f_miss: Optional[float] = None
    ancestry: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'variant_id': self.variant_id,
            'alt_freq': self.alt_freq,
            'obs_ct': self.obs_ct,
            'f_miss': self.f_miss,
            'ancestry': self.ancestry
        }


class VariantCollection:
    """Collection of variants with associated metadata."""
    
    def __init__(self):
        self.variants: Dict[str, Variant] = {}
        self.stats: Dict[str, List[VariantStats]] = {}
    
    def add_variant(self, variant: Variant):
        """Add a variant to the collection."""
        self.variants[variant.id] = variant
        
    def add_stats(self, stats: VariantStats):
        """Add statistics for a variant."""
        if stats.variant_id not in self.stats:
            self.stats[stats.variant_id] = []
        self.stats[stats.variant_id].append(stats)
    
    def get_variant(self, variant_id: str) -> Optional[Variant]:
        """Get a variant by ID."""
        return self.variants.get(variant_id)
    
    def get_stats(self, variant_id: str) -> List[VariantStats]:
        """Get statistics for a variant."""
        return self.stats.get(variant_id, [])
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        rows = []
        for variant_id, variant in self.variants.items():
            row = variant.to_dict()
            
            # Add aggregated statistics if available
            stats_list = self.get_stats(variant_id)
            if stats_list:
                # Calculate mean frequencies across ancestries
                alt_freqs = [s.alt_freq for s in stats_list if s.alt_freq is not None]
                if alt_freqs:
                    row['mean_alt_freq'] = sum(alt_freqs) / len(alt_freqs)
                
                # Add ancestry-specific frequencies
                for stat in stats_list:
                    if stat.ancestry:
                        row[f'alt_freq_{stat.ancestry}'] = stat.alt_freq
                        row[f'f_miss_{stat.ancestry}'] = stat.f_miss
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def from_dataframe(self, df: pd.DataFrame):
        """Load variants from a pandas DataFrame."""
        # Required columns for Variant
        required_cols = ['id', 'chrom', 'pos', 'ref', 'alt']
        
        for _, row in df.iterrows():
            # Create variant
            variant_data = {}
            for col in required_cols:
                if col in row:
                    variant_data[col] = row[col]
            
            # Add optional columns
            for col in ['rsid', 'hg38', 'hg19', 'locus', 'snp_name']:
                if col in row and pd.notna(row[col]):
                    variant_data[col] = row[col]
            
            variant = Variant(**variant_data)
            self.add_variant(variant)
            
            # Extract statistics columns
            for col in df.columns:
                if col.startswith('alt_freq_') and '_' in col[9:]:
                    ancestry = col.replace('alt_freq_', '')
                    if pd.notna(row[col]):
                        stats = VariantStats(
                            variant_id=variant.id,
                            alt_freq=row[col],
                            ancestry=ancestry
                        )
                        self.add_stats(stats)
