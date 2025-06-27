#!/usr/bin/env python3
"""
GP2 Precision Medicine Data Browser - Component Demonstration
============================================================

This script demonstrates how each component in the package works,
from basic low-level usage to advanced integration.

Run this to understand the current state of our development.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*70)
print("GP2 PRECISION MEDICINE DATA BROWSER - COMPONENT DEMONSTRATION")
print("="*70)

# ============================================================================
# LEVEL 1: CONFIGURATION SYSTEM (Foundation Layer)
# ============================================================================
print("\n🔧 LEVEL 1: CONFIGURATION SYSTEM")
print("-" * 50)

from src.core.config import settings

print("Configuration demonstrates environment-aware data path management:")
print(f"✓ Data root: {settings.data_root}")
print(f"✓ Clinical data directory: {settings.clinical_data_dir}")
print(f"✓ WGS data directory: {settings.wgs_data_dir}")
print(f"✓ NBA data directory: {settings.nba_data_dir}")

print("\nFull file paths (automatically computed):")
print(f"  - Clinical master key: {settings.clinical_master_key_path}")
print(f"  - WGS variant info: {settings.wgs_var_info_path}")
print(f"  - NBA variant info: {settings.nba_info_path}")

print(f"\nCache enabled: {settings.enable_cache}")
print("💡 The config system provides production-ready path management!")

# ============================================================================
# LEVEL 2: DATA MODELS (Pydantic Schemas)
# ============================================================================
print("\n\n📋 LEVEL 2: DATA MODELS (Pydantic Schemas)")
print("-" * 50)

from src.models.variant import VariantInfo, FilterCriteria
from src.models.clinical import ClinicalMetadata, ClinicalFilterCriteria
from src.models.sample import SampleCarrier, SampleFilterCriteria

print("Data models provide type-safe, validated data structures:")

# Example 1: Creating a variant info object
print("\n1. VariantInfo Model:")
example_variant = VariantInfo(
    id="chr1:12345_A_G",
    snp_name="rs123456",
    locus="GBA1",
    hg38="chr1:12345:A:G",
    hg19="chr1:12300:A:G", 
    chrom="1",
    pos=12345,
    a1="A",  # Reference allele
    a2="G"   # Alternative allele
)
print(f"   ✓ Created variant: {example_variant.id}")
print(f"   ✓ Location: {example_variant.chrom}:{example_variant.pos}")
print(f"   ✓ Gene: {example_variant.locus}")
print(f"   ✓ Alleles: {example_variant.a1} → {example_variant.a2}")
print(f"   ✓ Coordinates: hg38={example_variant.hg38}, hg19={example_variant.hg19}")

# Example 2: Creating filter criteria
print("\n2. FilterCriteria Model:")
variant_filters = FilterCriteria(
    loci=["GBA1", "LRRK2"],
    limit=10,
    offset=0
)
print(f"   ✓ Filter by genes: {variant_filters.loci}")
print(f"   ✓ Limit results: {variant_filters.limit}")
print("   💡 Pydantic automatically validates types and constraints!")

# Example 3: Clinical metadata
print("\n3. ClinicalMetadata Model:")
clinical_data = ClinicalMetadata(
    GP2ID="GP2_001234",  # Using alias
    study="PPMI",
    nba=0,  # Required field (0=no NBA data)
    wgs=1,  # Required field (1=has WGS data)
    clinical_exome=0,  # Required field
    extended_clinical_data=1,  # Required field
    GDPR=1,  # Required field (using alias)
    diagnosis="Parkinson Disease",
    wgs_label="European",  # WGS ancestry label
    nba_label=None  # No NBA label since nba=0
)
print(f"   ✓ Sample: {clinical_data.gp2_id}")
print(f"   ✓ Study: {clinical_data.study}")
print(f"   ✓ Ancestry: {clinical_data.primary_ancestry_label}")
print(f"   ✓ WGS available: {clinical_data.has_wgs_data}")
print(f"   ✓ Data source: {clinical_data.data_source}")

# ============================================================================
# LEVEL 3: REPOSITORY PATTERN (Data Access Layer)
# ============================================================================
print("\n\n🗄️ LEVEL 3: REPOSITORY PATTERN (Data Access Layer)")
print("-" * 50)

from src.repositories import VariantRepository, ClinicalRepository, SampleRepository

print("Repositories provide abstracted, lazy-loaded data access:")

# Example 1: Variant Repository
print("\n1. Variant Repository (WGS data):")
try:
    variant_repo = VariantRepository(data_source="WGS")
    print(f"   ✓ Created WGS variant repository")
    print(f"   ✓ Data loaded: {variant_repo.is_loaded}")
    print(f"   ✓ Data source: WGS")
    
    # Check if data files exist
    if variant_repo.data_path and variant_repo.data_path.exists():
        print(f"   ✓ Data file found: {variant_repo.data_path}")
        
        # Demonstrate lazy loading
        print("\n   Loading data (this may take a moment)...")
        variant_repo.load()
        total_variants = len(variant_repo)
        print(f"   ✓ Loaded {total_variants:,} variants")
        
        # Get available loci
        loci = variant_repo.get_loci()
        print(f"   ✓ Available genes: {len(loci)} (e.g., {loci[:5]})")
        
        # Demonstrate filtering
        gba1_filters = FilterCriteria(loci=["GBA1"], limit=3)
        gba1_variants = variant_repo.filter(gba1_filters)
        print(f"   ✓ GBA1 variants found: {len(gba1_variants)}")
        if gba1_variants:
            print(f"     Example: {gba1_variants[0].id} - {gba1_variants[0].snp_name}")
            
    else:
        print(f"   ⚠️ Data file not found: {variant_repo.data_path}")
        
except Exception as e:
    print(f"   ❌ Repository demo failed: {e}")

# Example 2: Clinical Repository
print("\n2. Clinical Repository:")
try:
    clinical_repo = ClinicalRepository()
    print(f"   ✓ Created clinical repository")
    
    if clinical_repo.data_path and clinical_repo.data_path.exists():
        clinical_repo.load()
        total_samples = len(clinical_repo)
        print(f"   ✓ Loaded {total_samples:,} clinical records")
        
        # Get unique studies
        studies = clinical_repo.get_studies()
        print(f"   ✓ Studies available: {len(studies)} (e.g., {studies[:3]})")
        
        # Get ancestry labels
        ancestries = clinical_repo.get_all_ancestry_labels()
        print(f"   ✓ Ancestry groups: {len(ancestries)} (e.g., {ancestries[:3]})")
        
    else:
        print(f"   ⚠️ Clinical data file not found: {clinical_repo.data_path}")
        
except Exception as e:
    print(f"   ❌ Clinical repository demo failed: {e}")

# Example 3: Sample Repository (Integration)
print("\n3. Sample Repository (Integrates clinical + carrier data):")
try:
    sample_repo = SampleRepository()
    print(f"   ✓ Created sample repository")
    
    # Get summary without loading all data
    summary = sample_repo.get_sample_summary()
    print(f"   ✓ Total samples tracked: {summary.total_samples:,}")
    print(f"   ✓ Samples with clinical data: {summary.samples_with_clinical:,}")
    print(f"   ✓ Sample sources: {summary.samples_by_source}")
    
    # Demonstrate filtered retrieval
    print("\n   Getting sample subset...")
    wgs_samples = sample_repo.get_samples_by_data_source("WGS", limit=2)
    print(f"   ✓ Retrieved {len(wgs_samples)} WGS samples")
    
    for sample in wgs_samples:
        print(f"     - Sample {sample.sample_id}")
        print(f"       Data source: {sample.data_source}")
        print(f"       Has clinical: {sample.has_clinical_data}")
        print(f"       Variants carried: {sample.get_carrier_count()}")
        if sample.clinical:
            print(f"       Study: {sample.clinical.study}")
            print(f"       Ancestry: {sample.clinical.primary_ancestry_label}")
            
except Exception as e:
    print(f"   ❌ Sample repository demo failed: {e}")

print("\n💡 Repository pattern provides:")
print("   - Lazy loading (data loads only when needed)")
print("   - Caching (loaded data stays in memory)")
print("   - Filtering (efficient data subsetting)")
print("   - Type safety (all results are validated models)")

# ============================================================================
# LEVEL 4: API LAYER (FastAPI Integration)
# ============================================================================
print("\n\n🌐 LEVEL 4: API LAYER (FastAPI Integration)")
print("-" * 50)

print("FastAPI layer provides RESTful endpoints with automatic documentation:")

# Demonstrate dependency injection
from src.api.dependencies import get_variant_repository, get_clinical_repository

print("\n1. Dependency Injection System:")
print("   ✓ get_variant_repository() - Cached WGS variant repository")
print("   ✓ get_clinical_repository() - Cached clinical repository")
print("   ✓ get_sample_repository() - Cached sample repository")
print("   💡 @lru_cache() ensures single instances across API requests")

# Show available endpoints
print("\n2. Available API Endpoints:")
print("   ✓ GET /              - Root API information")
print("   ✓ GET /health        - Health check")
print("   ✓ GET /api/variants/ - List variants (with filtering)")
print("   ✓ GET /api/variants/loci - Get available genes")

print("\n3. API Features:")
print("   ✓ Automatic request validation (Pydantic)")
print("   ✓ Automatic response serialization")
print("   ✓ Interactive documentation (/docs)")
print("   ✓ CORS middleware (for frontend integration)")
print("   ✓ Error handling with proper HTTP status codes")

# ============================================================================
# LEVEL 5: INTEGRATION EXAMPLE (How it all works together)
# ============================================================================
print("\n\n🚀 LEVEL 5: INTEGRATION EXAMPLE")
print("-" * 50)

print("Complete workflow: Configuration → Models → Repository → API")

def demonstrate_complete_workflow():
    """Demonstrates the complete data flow from config to API response."""
    
    print("\n1. Configuration loads environment-specific paths")
    data_root = settings.data_root
    print(f"   → Data root: {data_root}")
    
    print("\n2. Repository uses config to locate data files")
    variant_repo = get_variant_repository()  # Uses dependency injection
    print(f"   → Repository created for: {variant_repo.data_source}")
    
    print("\n3. Repository loads and validates data into Pydantic models")
    if variant_repo.data_path and variant_repo.data_path.exists():
        if not variant_repo.is_loaded:
            print("   → Loading data...")
            variant_repo.load()
        
        print("\n4. API endpoint uses repository to filter and return data")
        # Simulate API endpoint logic
        filters = FilterCriteria(loci=["GBA1"], limit=2)
        variants = variant_repo.filter(filters)
        
        print(f"   → Filtered to {len(variants)} variants")
        for variant in variants:
            print(f"     • {variant.id} ({variant.locus})")
            
        print("\n5. FastAPI automatically serializes models to JSON")
        # This would happen automatically in real API responses
        sample_response = {
            "variants": [
                {
                    "id": v.id,
                    "locus": v.locus,
                    "consequence": v.consequence,
                    "hgvs_p": v.hgvs_p
                } for v in variants
            ],
            "total": len(variants)
        }
        print(f"   → JSON response ready: {len(sample_response['variants'])} variants")
        
    else:
        print("   ⚠️ Demo data not available")

try:
    demonstrate_complete_workflow()
except Exception as e:
    print(f"   ❌ Integration demo failed: {e}")

# ============================================================================
# SUMMARY: CURRENT DEVELOPMENT STATUS
# ============================================================================
print("\n\n📊 CURRENT DEVELOPMENT STATUS")
print("="*70)

print("\n✅ COMPLETED COMPONENTS:")
print("   🔧 Configuration system (production-ready)")
print("   📋 Data models (all Pydantic schemas)")
print("   🗄️ Repository pattern (full CRUD + filtering)")
print("   🧪 Unit tests (repository layer)")
print("   🌐 Basic FastAPI app (health + variants endpoints)")
print("   📚 Dependency injection (cached repositories)")
print("   🔒 CORS middleware (frontend integration ready)")

print("\n🔄 IN PROGRESS:")
print("   🌐 API endpoints (variants ✓, samples needed)")
print("   🐳 Docker containerization (pending)")

print("\n⏳ UPCOMING (Phase 2+):")
print("   🔍 Advanced filtering/querying endpoints")
print("   💻 Streamlit frontend")
print("   📥 Download system")
print("   🔐 Authentication")

print("\n🎯 PHASE 1 STATUS: ~85% Complete")
print("   Next: Add samples API endpoint, then Docker setup")

print("\n" + "="*70)
print("🚀 Ready to continue with remaining Phase 1 components!")
print("="*70)
