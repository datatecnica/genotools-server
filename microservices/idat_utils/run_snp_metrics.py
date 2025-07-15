#!/usr/bin/env python3
"""Command-line interface for SNP Metrics Processing

Run SNP metrics processing pipeline with configurable paths.
"""

import argparse
import logging
import sys
import shutil
from pathlib import Path

from snp_metrics.processor import SNPProcessor
from snp_metrics.config import ProcessorConfig, ConfigurationError


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Process SNP metrics from IDAT files to parquet format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input path arguments (only required for normal processing, not variant reference)
    parser.add_argument(
        '--barcode-path',
        type=Path,
        help='Path to barcode directory containing IDAT files'
    )
    
    parser.add_argument(
        '--dragen-path',
        type=Path,
        help='Path to DRAGEN executable'
    )
    
    parser.add_argument(
        '--bpm-path',
        type=Path,
        help='Path to BPM manifest file (.bpm)'
    )
    
    parser.add_argument(
        '--bpm-csv-path',
        type=Path,
        help='Path to BPM CSV manifest file (.csv)'
    )
    
    parser.add_argument(
        '--egt-path',
        type=Path,
        help='Path to EGT cluster file (.egt)'
    )
    
    parser.add_argument(
        '--ref-fasta-path',
        type=Path,
        help='Path to reference genome FASTA file'
    )
    
    # Output path arguments (only required for normal processing, not variant reference)
    parser.add_argument(
        '--gtc-path',
        type=Path,
        help='Output directory for GTC files'
    )
    
    parser.add_argument(
        '--vcf-path',
        type=Path,
        help='Output directory for VCF files'
    )
    
    parser.add_argument(
        '--metrics-path',
        type=Path,
        required=True,
        help='Output directory for SNP metrics parquet files'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-file',
        type=Path,
        help='Custom output path for parquet file (overrides default naming)'
    )
    
    parser.add_argument(
        '--num-threads',
        type=int,
        default=1,
        help='Number of threads to use for DRAGEN processing'
    )
    
    parser.add_argument(
        '--no-cleanup',
        dest='cleanup',
        action='store_false',
        default=True,
        help='Keep intermediate GTC and VCF files (default: cleanup enabled)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration and exit without processing'
    )
    
    parser.add_argument(
        '--create-variant-ref',
        action='store_true',
        help='Create variant reference file instead of processing samples (requires --vcf-file)'
    )
    
    parser.add_argument(
        '--vcf-file',
        type=Path,
        help='Single VCF file to use for variant reference creation'
    )
    
    return parser


def validate_args(args) -> None:
    """Validate command line arguments."""
    # Special validation for variant reference mode
    if args.create_variant_ref:
        if not args.vcf_file:
            raise ValueError("--vcf-file is required when using --create-variant-ref")
        if not args.vcf_file.exists():
            raise ValueError(f"VCF file not found: {args.vcf_file}")
        if not args.metrics_path:
            raise ValueError("--metrics-path is required when using --create-variant-ref")
        return  # Skip other validations for variant reference mode
    
    # For normal processing, check that all required arguments are provided
    required_args = [
        ('barcode-path', args.barcode_path),
        ('dragen-path', args.dragen_path),
        ('bpm-path', args.bpm_path),
        ('bpm-csv-path', args.bpm_csv_path),
        ('egt-path', args.egt_path),
        ('ref-fasta-path', args.ref_fasta_path),
        ('gtc-path', args.gtc_path),
        ('vcf-path', args.vcf_path),
        ('metrics-path', args.metrics_path)
    ]
    
    missing_args = []
    for name, value in required_args:
        if value is None:
            missing_args.append(f'--{name}')
    
    if missing_args:
        raise ValueError(f"The following arguments are required for normal processing: {', '.join(missing_args)}")
    
    # Check that required input paths exist for normal processing
    required_paths = [
        ('barcode-path', args.barcode_path),
        ('dragen-path', args.dragen_path),
        ('bpm-path', args.bpm_path),
        ('bpm-csv-path', args.bpm_csv_path),
        ('egt-path', args.egt_path),
        ('ref-fasta-path', args.ref_fasta_path)
    ]
    
    missing_paths = []
    for name, path in required_paths:
        if not path.exists():
            missing_paths.append(f"--{name}: {path}")
    
    if missing_paths:
        print("ERROR: Required input paths not found:", file=sys.stderr)
        for path in missing_paths:
            print(f"  {path}", file=sys.stderr)
        sys.exit(1)
    
    # Validate num_threads
    if args.num_threads < 1:
        print("ERROR: Number of threads must be positive", file=sys.stderr)
        sys.exit(1)


def print_config(config: ProcessorConfig, barcode: str = None, output_file: Path = None, cleanup: bool = True, num_threads: int = 1, vcf_file: Path = None, variant_ref_mode: bool = False):
    """Print configuration summary."""
    if variant_ref_mode:
        print("🧬 Variant Reference Creation Configuration")
        print("=" * 50)
        print(f"VCF Input File: {vcf_file}")
        print(f"Metrics Output: {config.metrics_path}")
        if output_file:
            print(f"Custom Output Path: {output_file}")
        else:
            print(f"Default Output Path: {config.metrics_path}/variant_reference")
        print("=" * 50)
    else:
        print("🧬 SNP Metrics Processing Configuration")
        print("=" * 50)
        print(f"Barcode: {barcode} (extracted from path)")
        print(f"Barcode Path: {config.barcode_path}")
        print(f"DRAGEN: {config.dragen_path}")
        print(f"BPM Manifest: {config.bpm_path}")
        print(f"BPM CSV: {config.bpm_csv_path}")
        print(f"EGT Cluster: {config.egt_path}")
        print(f"Reference FASTA: {config.ref_fasta_path}")
        print(f"GTC Output: {config.gtc_path}")
        print(f"VCF Output: {config.vcf_path}")
        print(f"Metrics Output: {config.metrics_path}")
        print(f"Number of Threads: {num_threads}")
        print(f"Cleanup Intermediate Files: {'Yes' if cleanup else 'No'}")
        
        if output_file:
            print(f"Custom Output File: {output_file}")
        else:
            print(f"Default Output File: {config.metrics_path}/{barcode}")
        
        print("=" * 50)


def cleanup_intermediate_files(config: ProcessorConfig, barcode: str, logger):
    """Clean up intermediate GTC and VCF files."""
    gtc_dir = config.gtc_path / barcode
    vcf_dir = config.vcf_path / barcode
    
    dirs_to_clean = [
        ("GTC", gtc_dir),
        ("VCF", vcf_dir)
    ]
    
    for file_type, dir_path in dirs_to_clean:
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                logger.info(f"🗑️  Cleaned up {file_type} directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {file_type} directory {dir_path}: {e}")
        else:
            logger.debug(f"{file_type} directory not found: {dir_path}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    validate_args(args)
    
    try:
        if args.create_variant_ref:
            # For variant reference mode, just print config and skip ProcessorConfig creation
            print("🧬 Variant Reference Creation Configuration")
            print("=" * 50)
            print(f"VCF Input File: {args.vcf_file}")
            print(f"Metrics Output: {args.metrics_path}")
            if args.output_file:
                print(f"Custom Output Path: {args.output_file}")
            else:
                print(f"Default Output Path: {args.metrics_path}/variant_reference")
            print("=" * 50)
            
            config = None  # Not needed for variant reference
            barcode = None  # Not applicable for variant reference
        else:
            # Extract barcode from the barcode path (directory name)
            barcode = args.barcode_path.name
            logger.info(f"Extracted barcode '{barcode}' from path: {args.barcode_path}")
            
            # Create configuration for normal processing
            config = ProcessorConfig(
                barcode_path=args.barcode_path,
                dragen_path=args.dragen_path,
                bpm_path=args.bpm_path,
                bpm_csv_path=args.bpm_csv_path,
                egt_path=args.egt_path,
                ref_fasta_path=args.ref_fasta_path,
                gtc_path=args.gtc_path,
                vcf_path=args.vcf_path,
                metrics_path=args.metrics_path
            )
            
            # Print configuration for normal processing
            print_config(config, barcode, args.output_file, args.cleanup, args.num_threads)
        
        # Dry run - just show config and exit
        if args.dry_run:
            print("✅ Dry run complete - configuration is valid")
            return 0
        
        # Handle variant reference creation mode
        if args.create_variant_ref:
            logger.info(f"Creating variant reference from: {args.vcf_file}")
            
            # For variant reference, create processor with minimal config
            from snp_metrics.vcf_parser import VCFParser
            
            # Parse variant information directly
            parser = VCFParser()
            df = parser.parse_variants(str(args.vcf_file))
            
            # Write variant reference to parquet
            if args.output_file:
                output_path = Path(args.output_file)
            else:
                output_path = args.metrics_path / "variant_reference"
                
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if output already exists
            if output_path.exists():
                raise ValueError(
                    f"Output already exists at: {output_path}\n"
                    f"Please remove the existing file/directory before rerunning"
                )
            
            # Write with partitioning by chromosome only
            df.to_parquet(
                output_path, 
                compression='brotli', 
                index=False,
                partition_cols=['chromosome']
            )
            
            unique_chromosomes = df['chromosome'].nunique()
            total_variants = len(df)
            logger.info(f"Wrote {total_variants} variants to {output_path} "
                       f"({unique_chromosomes} chromosomes)")
            
            output_file = str(output_path)
            
            print(f"\n🎉 Variant Reference Creation Complete!")
            print(f"📁 Output saved to: {output_file}")
            
            # Show basic stats about the output
            try:
                import pandas as pd
                # Read the full partitioned dataset to get all columns including partition columns
                df_full = pd.read_parquet(output_file)
                print(f"📊 Variant reference contains columns: {list(df_full.columns)}")
                print(f"🧬 Partitioned by chromosome ({unique_chromosomes} chromosomes)")
                print(f"🔢 Total variants: {total_variants:,}")
                
                # Show sample of chromosome distribution
                if 'chromosome' in df_full.columns:
                    chr_dist = df_full['chromosome'].value_counts().head(5)
                    print(f"📈 Top chromosomes: {dict(chr_dist)}")
                    
            except Exception as e:
                logger.warning(f"Could not read output stats: {e}")
            
            return 0
        
        # Create processor and run normal sample processing
        logger.info(f"Starting SNP metrics processing for barcode: {barcode}")
        processor = SNPProcessor(config, num_threads=args.num_threads)
        
        # Process the barcode
        output_file = processor.process_barcode(
            barcode=barcode,
            output_path=str(args.output_file) if args.output_file else None
        )
        
        print(f"\n🎉 Processing Complete!")
        print(f"📁 Output saved to: {output_file}")
        
        # Cleanup intermediate files if requested
        if args.cleanup:
            logger.info("Cleaning up intermediate files...")
            cleanup_intermediate_files(config, barcode, logger)
            print("🗑️  Intermediate files cleaned up")
        else:
            logger.info("Skipping cleanup - intermediate files preserved")
            print("📂 Intermediate files preserved")
        
        # Show basic stats about the output
        try:
            import pandas as pd
            df = pd.read_parquet(output_file)
            print(f"📊 Generated {len(df):,} SNP records")
            print(f"🧬 Columns: {list(df.columns)}")
            
            # Show sample of genotype distribution if available
            if 'GT' in df.columns:
                gt_counts = df['GT'].value_counts().sort_index()
                print(f"🔢 Genotype distribution:")
                gt_labels = {0: '0/0 (homozygous ref)', 1: '0/1 (heterozygous)', 
                           2: '1/1 (homozygous alt)', -9: './. (no call)'}
                for gt, count in gt_counts.items():
                    label = gt_labels.get(gt, f'GT={gt}')
                    print(f"   {label}: {count:,}")
                    
        except Exception as e:
            logger.warning(f"Could not read output stats: {e}")
        
        return 0
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 