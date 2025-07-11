"""Example usage of the SNP Metrics Processor

Demonstrates how to process a single barcode using the clean API.
"""

import logging
from pathlib import Path

from snp_metrics.processor import SNPProcessor
from snp_metrics.config import ProcessorConfig


def main():
    """Example of processing a single barcode."""
    
    # Set up logging to see processing progress
    logging.basicConfig(level=logging.INFO)
    
    # Configure paths explicitly - update these for your environment
    data_path = Path("/home/vitaled2/genotools-server/microservices/idat_utils/data")
    barcode = "205746280003"
    
    config = ProcessorConfig(
        barcode_path=data_path / "idats" / barcode,
        dragen_path=data_path.parent / "dragena-linux-x64-DAv1.2.0-rc3-sha.3c7ece3a88eeeff572c7c97ab39da980714335c0" / "dragena" / "dragena",
        bpm_path=data_path / "ilmn_utils" / "NeuroBooster_20042459_A2.bpm",
        bpm_csv_path=data_path / "ilmn_utils" / "NeuroBooster_20042459_A2.csv",
        egt_path=data_path / "ilmn_utils" / "recluster_09092022.egt",
        ref_fasta_path=data_path / "ref" / "GRCh38_genome.fa",
        gtc_path=data_path / "output" / "gtcs",
        vcf_path=data_path / "output" / "vcfs",
        metrics_path=data_path / "output" / "snp_metrics"
    )
    
    # Create processor
    processor = SNPProcessor(config)
    
    try:
        output_file = processor.process_barcode(barcode)
        print(f"✅ Successfully processed {barcode}")
        print(f"📁 Output saved to: {output_file}")
        
        # Optionally, verify the output
        import pandas as pd
        df = pd.read_parquet(output_file)
        print(f"📊 Generated {len(df)} SNP records with {len(df.columns)} columns")
        print(f"🧬 Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")


def process_with_custom_config():
    """Example using custom configuration."""
    
    # Create config with custom paths - point directly to barcode directory
    config = ProcessorConfig(
        barcode_path=Path("/path/to/idats/your_barcode_here"),
        dragen_path=Path("/path/to/dragen/dragena"),
        bpm_path=Path("/path/to/manifest.bpm"),
        bpm_csv_path=Path("/path/to/manifest.csv"),
        egt_path=Path("/path/to/cluster.egt"),
        ref_fasta_path=Path("/path/to/reference.fa"),
        gtc_path=Path("/path/to/output/gtcs"),
        vcf_path=Path("/path/to/output/vcfs"),
        metrics_path=Path("/path/to/output/metrics")
    )
    
    processor = SNPProcessor(config)
    
    # Process with custom output path
    output_file = processor.process_barcode(
        barcode="your_barcode_here",
        output_path="/custom/path/to/output.parquet"
    )
    
    return output_file


if __name__ == "__main__":
    main() 