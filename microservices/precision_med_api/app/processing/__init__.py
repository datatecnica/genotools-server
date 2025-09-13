"""
Genomic variant processing and extraction module.

This module provides functionality for:
- Real-time variant harmonization
- Extracting variants from PLINK files with allele harmonization
- Transforming genotypes to match reference orientation
- Outputting results in standard formats (TRAW, Parquet)
- Coordinating multi-source extractions

Main components:
- extractor: Variant extraction from PLINK files
- transformer: Genotype transformation logic
- output: Output formatting (TRAW, reports)
- coordinator: High-level extraction orchestration
"""

from .extractor import VariantExtractor
from .harmonizer import HarmonizationEngine
from .transformer import GenotypeTransformer
from .output import TrawFormatter
from .coordinator import ExtractionCoordinator

__all__ = [
    "VariantExtractor",
    "HarmonizationEngine",
    "GenotypeTransformer",
    "TrawFormatter",
    "ExtractionCoordinator"
]