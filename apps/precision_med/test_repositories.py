#!/usr/bin/env python3
"""
Test script for the GP2 Precision Medicine Data Browser repository implementation.
This script validates that the repository pattern works correctly with actual data.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import settings
from src.repositories import VariantRepository, ClinicalRepository, SampleRepository
from src.models.variant import FilterCriteria
from src.models.clinical import ClinicalFilterCriteria
from src.models.sample import SampleFilterCriteria
from src.utils.data_utils import validate_data_paths, get_csv_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_data_paths():
    """Test that all data paths are accessible."""
    logger.info("Testing data paths...")
    
    path_results = validate_data_paths(settings)
    
    for name, exists in path_results.items():
        status = "✓" if exists else "✗"
        logger.info(f"  {status} {name}: {exists}")
    
    return all(path_results.values())


def test_data_file_info():
    """Get information about data files."""
    logger.info("Analyzing data files...")
    
    files_to_check = [
        ("Clinical master key", settings.clinical_master_key_path),
        ("WGS variant info", settings.wgs_var_info_path),
        ("NBA variant info", settings.nba_info_path),
    ]
    
    for name, path in files_to_check:
        logger.info(f"\n{name}:")
        info = get_csv_info(path)
        
        if info.get("exists"):
            logger.info(f"  Size: {info['file_size_mb']:.2f} MB")
            logger.info(f"  Estimated memory: {info['estimated_memory_mb']:.2f} MB")
            logger.info(f"  Rows: {info.get('total_rows', 'Unknown'):,}")
            logger.info(f"  Columns: {info.get('total_columns', 0)}")
        else:
            logger.warning(f"  File not found")


def test_variant_repository():
    """Test variant repository functionality."""
    logger.info("\nTesting variant repository...")
    
    try:
        # Test WGS variant repository
        variant_repo = VariantRepository(data_source="WGS")
        
        # Load a small sample
        logger.info("Loading variant data...")
        variant_repo.load()
        
        # Get basic stats
        total_variants = len(variant_repo)
        logger.info(f"Total WGS variants: {total_variants:,}")
        
        # Get unique loci
        loci = variant_repo.get_loci()
        logger.info(f"Unique loci: {len(loci)}")
        logger.info(f"Sample loci: {loci[:5]}")
        
        # Test filtering
        logger.info("Testing variant filtering...")
        filters = FilterCriteria(loci=["GBA1"], limit=5)
        gba1_variants = variant_repo.filter(filters)
        logger.info(f"GBA1 variants found: {len(gba1_variants)}")
        
        if gba1_variants:
            sample_variant = gba1_variants[0]
            logger.info(f"Sample variant: {sample_variant.id} - {sample_variant.snp_name}")
        
        logger.info("✓ Variant repository test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Variant repository test failed: {e}")
        return False


def test_clinical_repository():
    """Test clinical repository functionality."""
    logger.info("\nTesting clinical repository...")
    
    try:
        clinical_repo = ClinicalRepository()
        
        # Load data
        logger.info("Loading clinical data...")
        clinical_repo.load()
        
        # Get basic stats
        total_samples = len(clinical_repo)
        logger.info(f"Total clinical samples: {total_samples:,}")
        
        # Get studies
        studies = clinical_repo.get_studies()
        logger.info(f"Unique studies: {len(studies)}")
        logger.info(f"Sample studies: {studies[:5]}")
        
        # Get ancestry labels
        ancestry_labels = clinical_repo.get_all_ancestry_labels()
        logger.info(f"Ancestry labels: {len(ancestry_labels)}")
        logger.info(f"Sample ancestries: {ancestry_labels[:5]}")
        
        # Test filtering
        logger.info("Testing clinical filtering...")
        filters = ClinicalFilterCriteria(has_wgs=True, limit=5)
        wgs_samples = clinical_repo.filter(filters)
        logger.info(f"Samples with WGS data: {len(wgs_samples)}")
        
        if wgs_samples:
            sample = wgs_samples[0]
            logger.info(f"Sample: {sample.gp2_id} - {sample.study} - {sample.primary_ancestry_label}")
        
        logger.info("✓ Clinical repository test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Clinical repository test failed: {e}")
        return False


def test_sample_repository():
    """Test sample repository functionality."""
    logger.info("\nTesting sample repository...")
    
    try:
        sample_repo = SampleRepository()
        
        # Get summary
        logger.info("Getting sample summary...")
        summary = sample_repo.get_sample_summary()
        
        logger.info(f"Total samples: {summary.total_samples:,}")
        logger.info(f"Samples with clinical data: {summary.samples_with_clinical:,}")
        logger.info(f"Samples by source: {summary.samples_by_source}")
        
        # Test getting samples by data source
        logger.info("Testing sample retrieval...")
        wgs_samples = sample_repo.get_samples_by_data_source("WGS", limit=3)
        logger.info(f"Retrieved {len(wgs_samples)} WGS samples")
        
        if wgs_samples:
            sample = wgs_samples[0]
            logger.info(f"Sample: {sample.sample_id}")
            logger.info(f"  Data source: {sample.data_source}")
            logger.info(f"  Has clinical: {sample.has_clinical_data}")
            logger.info(f"  Variants carried: {sample.get_carrier_count()}")
            
            if sample.clinical:
                logger.info(f"  Study: {sample.clinical.study}")
                logger.info(f"  Ancestry: {sample.clinical.primary_ancestry_label}")
        
        sample_repo.clear_cache()
        logger.info("✓ Sample repository test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Sample repository test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting GP2 Precision Medicine Data Browser repository tests")
    logger.info("=" * 60)
    
    tests = [
        ("Data paths validation", test_data_paths),
        ("Data file analysis", test_data_file_info),
        ("Variant repository", test_variant_repository),
        ("Clinical repository", test_clinical_repository),
        ("Sample repository", test_sample_repository),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'-' * 40}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'-' * 40}")
        
        try:
            if test_name == "Data file analysis":
                test_func()  # This test doesn't return a boolean
                results.append((test_name, True))
            else:
                result = test_func()
                results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        logger.info(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Repository pattern implementation is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 