"""
Genotype analysis utility functions for carrier identification and statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def identify_carriers(genotype_data: Dict[str, Any], threshold: int = 0) -> Dict[str, Any]:
    """
    Identify carrier samples across data types.

    Args:
        genotype_data: Output from GenotypeDataLoader
        threshold: Minimum genotype value to consider as carrier (default: 0, so >0 = carrier)

    Returns:
        Dictionary with carrier information per data type and overall
    """
    if 'error' in genotype_data:
        return {'error': genotype_data['error']}

    carrier_info = {
        'by_data_type': {},
        'summary': {
            'total_unique_carriers': 0,
            'carriers_in_multiple_types': 0,
            'data_type_overlap': {}
        }
    }

    all_carriers = set()

    for data_type, data in genotype_data['data'].items():
        genotypes = data['genotypes']

        # Find samples with any carrier genotype (> threshold)
        carrier_mask = genotypes > threshold
        carrier_variants = carrier_mask.any(axis=1)
        carrier_samples = carrier_mask.any(axis=0)

        # Get carrier sample names
        carrier_sample_names = [col for col, is_carrier in carrier_samples.items() if is_carrier]

        # Per-variant carrier counts
        variant_carrier_counts = carrier_mask.sum(axis=1)

        carrier_info['by_data_type'][data_type] = {
            'carrier_samples': carrier_sample_names,
            'carrier_count': len(carrier_sample_names),
            'total_samples': len(data['sample_columns']),
            'carrier_frequency': len(carrier_sample_names) / len(data['sample_columns']) if data['sample_columns'] else 0,
            'variants_with_carriers': carrier_variants.sum(),
            'total_variants': len(genotypes),
            'per_variant_carrier_counts': variant_carrier_counts.to_dict()
        }

        all_carriers.update(carrier_sample_names)

    # Calculate cross-data-type statistics
    carrier_info['summary']['total_unique_carriers'] = len(all_carriers)

    # Find overlaps between data types
    data_types = list(genotype_data['data'].keys())
    for i, dt1 in enumerate(data_types):
        for dt2 in data_types[i+1:]:
            if dt1 in carrier_info['by_data_type'] and dt2 in carrier_info['by_data_type']:
                carriers1 = set(carrier_info['by_data_type'][dt1]['carrier_samples'])
                carriers2 = set(carrier_info['by_data_type'][dt2]['carrier_samples'])
                overlap = carriers1.intersection(carriers2)

                carrier_info['summary']['data_type_overlap'][f"{dt1}_vs_{dt2}"] = {
                    'overlap_count': len(overlap),
                    'overlap_samples': list(overlap),
                    'dt1_only': list(carriers1 - carriers2),
                    'dt2_only': list(carriers2 - carriers1)
                }

    return carrier_info


def calculate_carrier_frequencies(genotype_data: Dict[str, Any], by_variant: bool = True,
                                by_gene: bool = True) -> Dict[str, Any]:
    """
    Calculate carrier frequencies at different levels.

    Args:
        genotype_data: Output from GenotypeDataLoader
        by_variant: Calculate per-variant frequencies
        by_gene: Calculate per-gene frequencies

    Returns:
        Dictionary with frequency calculations
    """
    if 'error' in genotype_data:
        return {'error': genotype_data['error']}

    frequencies = {
        'overall': {},
        'by_variant': {},
        'by_gene': {}
    }

    for data_type, data in genotype_data['data'].items():
        genotypes = data['genotypes']
        metadata = data['metadata']
        sample_count = len(data['sample_columns'])

        # Overall frequency for this data type
        total_carriers = (genotypes > 0).sum().sum()
        total_possible = len(genotypes) * sample_count
        overall_freq = total_carriers / total_possible if total_possible > 0 else 0

        frequencies['overall'][data_type] = {
            'carrier_frequency': overall_freq,
            'total_carriers': int(total_carriers),
            'total_genotypes': int(total_possible),
            'sample_count': sample_count,
            'variant_count': len(genotypes)
        }

        if by_variant:
            variant_frequencies = {}
            for idx in genotypes.index:
                variant_carriers = (genotypes.loc[idx] > 0).sum()
                variant_freq = variant_carriers / sample_count if sample_count > 0 else 0

                # Get variant identifier
                if 'variant_id' in metadata.columns:
                    variant_id = metadata.loc[idx, 'variant_id']
                else:
                    variant_id = f"variant_{idx}"

                variant_frequencies[variant_id] = {
                    'carrier_frequency': variant_freq,
                    'carrier_count': int(variant_carriers),
                    'sample_count': sample_count
                }

            frequencies['by_variant'][data_type] = variant_frequencies

        if by_gene and 'locus' in metadata.columns:
            gene_frequencies = {}
            for gene in metadata['locus'].dropna().unique():
                gene_mask = metadata['locus'] == gene
                gene_genotypes = genotypes[gene_mask]

                if not gene_genotypes.empty:
                    # Count samples that are carriers for ANY variant in this gene
                    gene_carriers = (gene_genotypes > 0).any(axis=0).sum()
                    gene_freq = gene_carriers / sample_count if sample_count > 0 else 0

                    gene_frequencies[gene] = {
                        'carrier_frequency': gene_freq,
                        'carrier_count': int(gene_carriers),
                        'sample_count': sample_count,
                        'variant_count': len(gene_genotypes)
                    }

            frequencies['by_gene'][data_type] = gene_frequencies

    return frequencies


def compare_data_types(genotype_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare carrier detection across different data types.

    Args:
        genotype_data: Output from GenotypeDataLoader

    Returns:
        Dictionary with comparison statistics
    """
    if 'error' in genotype_data:
        return {'error': genotype_data['error']}

    data_types = list(genotype_data['data'].keys())
    if len(data_types) < 2:
        return {'error': 'Need at least 2 data types for comparison'}

    comparison = {
        'data_types': data_types,
        'sample_overlap': {},
        'variant_overlap': {},
        'carrier_concordance': {}
    }

    # Get sample overlaps
    all_samples = {}
    for dt in data_types:
        all_samples[dt] = set(genotype_data['data'][dt]['sample_columns'])

    for i, dt1 in enumerate(data_types):
        for dt2 in data_types[i+1:]:
            samples1 = all_samples[dt1]
            samples2 = all_samples[dt2]
            overlap = samples1.intersection(samples2)

            comparison['sample_overlap'][f"{dt1}_vs_{dt2}"] = {
                'shared_samples': len(overlap),
                'dt1_total': len(samples1),
                'dt2_total': len(samples2),
                'overlap_percentage': len(overlap) / min(len(samples1), len(samples2)) * 100 if samples1 and samples2 else 0
            }

            # Carrier concordance for shared samples
            if overlap:
                # Get genotype data for shared samples only
                shared_sample_list = list(overlap)

                dt1_genotypes = genotype_data['data'][dt1]['genotypes'][shared_sample_list]
                dt2_genotypes = genotype_data['data'][dt2]['genotypes'][shared_sample_list]

                # Find carriers in each dataset
                dt1_carriers = (dt1_genotypes > 0).any(axis=0)
                dt2_carriers = (dt2_genotypes > 0).any(axis=0)

                # Calculate concordance
                both_carriers = (dt1_carriers & dt2_carriers).sum()
                either_carrier = (dt1_carriers | dt2_carriers).sum()

                comparison['carrier_concordance'][f"{dt1}_vs_{dt2}"] = {
                    'both_carriers': int(both_carriers),
                    'dt1_only_carriers': int((dt1_carriers & ~dt2_carriers).sum()),
                    'dt2_only_carriers': int((dt2_carriers & ~dt1_carriers).sum()),
                    'neither_carrier': int((~dt1_carriers & ~dt2_carriers).sum()),
                    'concordance_rate': both_carriers / either_carrier if either_carrier > 0 else 0,
                    'shared_samples_analyzed': len(shared_sample_list)
                }

    return comparison


def generate_carrier_summary(genotype_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive carrier summary statistics.

    Args:
        genotype_data: Output from GenotypeDataLoader

    Returns:
        Dictionary with summary statistics ready for display
    """
    if 'error' in genotype_data:
        return {'error': genotype_data['error']}

    summary = {
        'overview': genotype_data['summary'],
        'filters_applied': genotype_data['filters_applied'],
        'by_data_type': {},
        'top_genes': [],
        'top_variants': []
    }

    # Detailed breakdown by data type
    for data_type, data in genotype_data['data'].items():
        dt_summary = {
            'basic_stats': data['stats'],
            'sample_info': {
                'total_samples': len(data['sample_columns']),
                'first_few_samples': data['sample_columns'][:5],
                'sample_id_pattern': _analyze_sample_id_pattern(data['sample_columns'])
            }
        }

        # Add gene-level stats if available
        if 'locus' in data['metadata'].columns:
            gene_stats = {}
            for gene in data['metadata']['locus'].dropna().unique():
                gene_mask = data['metadata']['locus'] == gene
                gene_genotypes = data['genotypes'][gene_mask]
                gene_carriers = (gene_genotypes > 0).any(axis=0).sum()

                gene_stats[gene] = {
                    'variant_count': gene_mask.sum(),
                    'carrier_count': int(gene_carriers),
                    'carrier_frequency': gene_carriers / len(data['sample_columns']) if data['sample_columns'] else 0
                }

            dt_summary['gene_stats'] = gene_stats

            # Top genes by carrier count
            top_genes = sorted(gene_stats.items(), key=lambda x: x[1]['carrier_count'], reverse=True)[:10]
            summary['top_genes'].extend([(data_type, gene, stats) for gene, stats in top_genes])

        summary['by_data_type'][data_type] = dt_summary

    # Sort top genes across all data types
    summary['top_genes'] = sorted(summary['top_genes'], key=lambda x: x[2]['carrier_count'], reverse=True)[:10]

    return summary


def _analyze_sample_id_pattern(sample_columns: List[str]) -> Dict[str, Any]:
    """Analyze sample ID patterns for QC purposes."""
    if not sample_columns:
        return {'pattern': 'No samples', 'examples': []}

    # Look for common patterns
    patterns = {
        'sample_prefix': any(col.startswith('SAMPLE_') for col in sample_columns[:10]),
        'numeric_suffix': any(col.split('_')[-1].isdigit() for col in sample_columns[:10] if '_' in col),
        'average_length': np.mean([len(col) for col in sample_columns]),
        'unique_prefixes': len(set(col.split('_')[0] for col in sample_columns[:100] if '_' in col))
    }

    return {
        'pattern_analysis': patterns,
        'examples': sample_columns[:5],
        'total_samples': len(sample_columns)
    }


def create_genotype_matrix_display(genotype_data: Dict[str, Any], max_variants: int = 20,
                                 max_samples: int = 20) -> Dict[str, pd.DataFrame]:
    """
    Create display-ready genotype matrices.

    Args:
        genotype_data: Output from GenotypeDataLoader
        max_variants: Maximum variants to show
        max_samples: Maximum samples to show

    Returns:
        Dictionary of DataFrames ready for display
    """
    if 'error' in genotype_data:
        return {'error': genotype_data['error']}

    display_matrices = {}

    for data_type, data in genotype_data['data'].items():
        genotypes = data['genotypes']
        metadata = data['metadata']

        # Limit size for display
        display_genotypes = genotypes.iloc[:max_variants, :max_samples].copy()
        display_metadata = metadata.iloc[:max_variants].copy() if not metadata.empty else pd.DataFrame(index=display_genotypes.index)

        # Create variant labels
        variant_labels = []
        for idx in display_genotypes.index:
            if 'variant_id' in display_metadata.columns:
                label = display_metadata.loc[idx, 'variant_id']
            elif 'locus' in display_metadata.columns and 'position' in display_metadata.columns:
                locus = display_metadata.loc[idx, 'locus'] if 'locus' in display_metadata.columns else 'Unknown'
                pos = display_metadata.loc[idx, 'position'] if 'position' in display_metadata.columns else 'Unknown'
                label = f"{locus}:{pos}"
            else:
                label = f"Variant_{idx}"
            variant_labels.append(label)

        # Create display DataFrame with variant labels as index
        display_df = display_genotypes.copy()
        display_df.index = variant_labels

        display_matrices[data_type] = display_df

    return display_matrices