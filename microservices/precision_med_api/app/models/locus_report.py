"""
Pydantic models for per-locus clinical phenotype reports.

Defines data structures for ancestry-stratified clinical metrics
aggregated by gene/locus for carrier screening analysis.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, computed_field


class VariantDetail(BaseModel):
    """Detailed information about a single variant including carrier counts."""

    variant_id: str = Field(..., description="Variant identifier (e.g., chr1:7965425:G:C)")
    mutation_name: str = Field("", description="Mutation name from SNP list (e.g., E64D, p.Leu10Pro)")
    chromosome: str = Field(..., description="Chromosome (e.g., chr1, 1)")
    position: int = Field(..., description="Genomic position")
    ref_allele: str = Field(..., description="Reference allele")
    alt_allele: str = Field(..., description="Alternate allele")
    carrier_count: int = Field(..., description="Total carriers (genotype > 0)")
    heterozygous_count: int = Field(0, description="Heterozygous carriers (genotype == 1)")
    homozygous_count: int = Field(0, description="Homozygous carriers (genotype == 2)")


class ClinicalMetrics(BaseModel):
    """Clinical phenotype counts for one ancestry group."""

    ancestry: str = Field(..., description="Genetic ancestry label (e.g., AJ, EUR, EAS)")

    # Carrier counts
    total_carriers: int = Field(..., description="Total carriers (genotype > 0)")
    carriers_with_clinical_data: int = Field(..., description="Carriers with extended clinical data available")

    # Hoehn & Yahr stage metrics
    hy_available: int = Field(0, description="Carriers with H&Y stage data")
    hy_less_than_2: int = Field(0, description="Carriers with H&Y < 2")
    hy_less_than_3: int = Field(0, description="Carriers with H&Y < 3")

    # MoCA (Montreal Cognitive Assessment) metrics
    moca_available: int = Field(0, description="Carriers with MoCA total score")
    moca_gte_20: int = Field(0, description="Carriers with MoCA ≥ 20")
    moca_gte_24: int = Field(0, description="Carriers with MoCA ≥ 24")

    # DAT scan metrics
    dat_caudate_available: int = Field(0, description="Carriers with DAT caudate mean available")

    # Disease duration metrics
    disease_duration_lte_3_years: int = Field(0, description="Carriers with disease duration ≤ 3 years")
    disease_duration_lte_5_years: int = Field(0, description="Carriers with disease duration ≤ 5 years")
    disease_duration_lte_7_years: int = Field(0, description="Carriers with disease duration ≤ 7 years")

    @computed_field
    @property
    def clinical_data_availability_pct(self) -> float:
        """Percentage of carriers with clinical data."""
        if self.total_carriers == 0:
            return 0.0
        return (self.carriers_with_clinical_data / self.total_carriers) * 100


class LocusReport(BaseModel):
    """Complete clinical phenotype report for one gene/locus."""

    locus: str = Field(..., description="Gene name (e.g., LRRK2, GBA1, PRKN)")

    # Metrics by ancestry
    by_ancestry: List[ClinicalMetrics] = Field(..., description="Metrics stratified by ancestry")

    # Aggregated totals
    total_metrics: ClinicalMetrics = Field(..., description="Total metrics across all ancestries")

    # Variant details with carrier counts
    variant_details: List[VariantDetail] = Field(default_factory=list, description="Detailed variant information with carrier counts")

    # Metadata (kept for backward compatibility)
    n_variants: int = Field(..., description="Number of variants in this locus")
    variant_ids: List[str] = Field(default_factory=list, description="Variant IDs included in this locus")

    @computed_field
    @property
    def ancestries_represented(self) -> List[str]:
        """List of ancestries with carriers in this locus."""
        return [m.ancestry for m in self.by_ancestry if m.total_carriers > 0]

    @computed_field
    @property
    def total_carriers_all_ancestries(self) -> int:
        """Total carriers across all ancestries."""
        return sum(m.total_carriers for m in self.by_ancestry)


class LocusReportSummary(BaseModel):
    """High-level summary statistics for locus report generation."""

    total_loci_analyzed: int = Field(..., description="Total number of genes/loci analyzed")
    total_carriers_identified: int = Field(..., description="Total carriers across all loci")
    total_samples_analyzed: int = Field(0, description="Total samples in this dataset")
    total_samples_with_clinical_data: int = Field(..., description="Total samples with extended clinical data")
    total_variants: int = Field(0, description="Total variants across all loci")

    # Ancestry representation
    ancestries_represented: List[str] = Field(default_factory=list, description="List of ancestries with carriers")
    carriers_by_ancestry: Dict[str, int] = Field(default_factory=dict, description="Total carriers per ancestry")


class LocusReportCollection(BaseModel):
    """Complete collection of locus reports for a pipeline run."""

    job_id: str = Field(..., description="Pipeline job identifier")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="When analysis was performed")

    # Configuration
    data_type: str = Field(..., description="Data type (WGS, NBA, or IMPUTED)")

    # Summary statistics
    summary: LocusReportSummary = Field(..., description="High-level summary statistics")

    # Per-locus reports
    locus_reports: List[LocusReport] = Field(..., description="Clinical reports for each locus")

    # Clinical data sources
    clinical_data_sources: Dict[str, str] = Field(
        default_factory=dict,
        description="Paths to clinical data files used"
    )

    @computed_field
    @property
    def loci_analyzed(self) -> List[str]:
        """List of loci included in this report collection."""
        return [report.locus for report in self.locus_reports]

    @computed_field
    @property
    def top_loci_by_carriers(self) -> List[Dict[str, Any]]:
        """Top 10 loci by total carrier count."""
        sorted_loci = sorted(
            self.locus_reports,
            key=lambda x: x.total_carriers_all_ancestries,
            reverse=True
        )[:10]
        return [
            {
                "locus": report.locus,
                "total_carriers": report.total_carriers_all_ancestries,
                "ancestries": len(report.ancestries_represented)
            }
            for report in sorted_loci
        ]
