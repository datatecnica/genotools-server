"""
Pydantic models for probe validation and selection analysis.

Contains models for both diagnostic classification and genotype concordance
approaches to probe quality assessment.
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, computed_field


class DiagnosticMetrics(BaseModel):
    """Traditional diagnostic test metrics treating carriers as positive cases."""

    true_positives: int = Field(..., description="NBA detects carrier, WGS confirms")
    false_positives: int = Field(..., description="NBA detects carrier, WGS shows non-carrier")
    false_negatives: int = Field(..., description="NBA misses carrier, WGS confirms carrier")
    true_negatives: int = Field(..., description="NBA and WGS both show non-carrier")

    @computed_field
    @property
    def sensitivity(self) -> float:
        """Ability to detect true carriers: TP/(TP+FN)"""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @computed_field
    @property
    def specificity(self) -> float:
        """Ability to avoid false alarms: TN/(TN+FP)"""
        denominator = self.true_negatives + self.false_positives
        return self.true_negatives / denominator if denominator > 0 else 0.0

    @computed_field
    @property
    def ppv(self) -> float:
        """Positive predictive value: TP/(TP+FP)"""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @computed_field
    @property
    def npv(self) -> float:
        """Negative predictive value: TN/(TN+FN)"""
        denominator = self.true_negatives + self.false_negatives
        return self.true_negatives / denominator if denominator > 0 else 0.0


class GenotypeTransitionMatrix(BaseModel):
    """Genotype transition matrix showing NBA→WGS transitions."""

    nba_0_wgs_0: int = Field(..., description="NBA=0, WGS=0 (correct non-carrier)")
    nba_0_wgs_1: int = Field(..., description="NBA=0, WGS=1 (missed heterozygous)")
    nba_0_wgs_2: int = Field(..., description="NBA=0, WGS=2 (missed homozygous)")
    nba_1_wgs_0: int = Field(..., description="NBA=1, WGS=0 (false heterozygous)")
    nba_1_wgs_1: int = Field(..., description="NBA=1, WGS=1 (correct heterozygous)")
    nba_1_wgs_2: int = Field(..., description="NBA=1, WGS=2 (het/hom misclassification)")
    nba_2_wgs_0: int = Field(..., description="NBA=2, WGS=0 (false homozygous)")
    nba_2_wgs_1: int = Field(..., description="NBA=2, WGS=1 (hom/het misclassification)")
    nba_2_wgs_2: int = Field(..., description="NBA=2, WGS=2 (correct homozygous)")


class ConcordanceMetrics(BaseModel):
    """Genotype concordance analysis with detailed error classification."""

    total_samples_compared: int = Field(..., description="Total samples with both NBA and WGS data")
    transition_matrix: GenotypeTransitionMatrix = Field(..., description="Full genotype transition matrix")

    @computed_field
    @property
    def overall_concordance(self) -> float:
        """Overall genotype concordance rate."""
        matrix = self.transition_matrix
        correct = matrix.nba_0_wgs_0 + matrix.nba_1_wgs_1 + matrix.nba_2_wgs_2
        return correct / self.total_samples_compared if self.total_samples_compared > 0 else 0.0

    @computed_field
    @property
    def wt_concordance(self) -> float:
        """Wild-type (0/0) concordance rate."""
        matrix = self.transition_matrix
        wgs_0_total = matrix.nba_0_wgs_0 + matrix.nba_1_wgs_0 + matrix.nba_2_wgs_0
        return matrix.nba_0_wgs_0 / wgs_0_total if wgs_0_total > 0 else 0.0

    @computed_field
    @property
    def het_concordance(self) -> float:
        """Heterozygous (0/1) concordance rate."""
        matrix = self.transition_matrix
        wgs_1_total = matrix.nba_0_wgs_1 + matrix.nba_1_wgs_1 + matrix.nba_2_wgs_1
        return matrix.nba_1_wgs_1 / wgs_1_total if wgs_1_total > 0 else 0.0

    @computed_field
    @property
    def hom_concordance(self) -> float:
        """Homozygous (1/1) concordance rate."""
        matrix = self.transition_matrix
        wgs_2_total = matrix.nba_0_wgs_2 + matrix.nba_1_wgs_2 + matrix.nba_2_wgs_2
        return matrix.nba_2_wgs_2 / wgs_2_total if wgs_2_total > 0 else 0.0

    @computed_field
    @property
    def false_negatives(self) -> int:
        """Missed carriers: NBA=0, WGS=1|2"""
        return self.transition_matrix.nba_0_wgs_1 + self.transition_matrix.nba_0_wgs_2

    @computed_field
    @property
    def false_positives(self) -> int:
        """False carriers: NBA=1|2, WGS=0"""
        return self.transition_matrix.nba_1_wgs_0 + self.transition_matrix.nba_2_wgs_0

    @computed_field
    @property
    def genotype_misclassification(self) -> int:
        """Het/Hom misclassification: NBA=1,WGS=2 or NBA=2,WGS=1"""
        return self.transition_matrix.nba_1_wgs_2 + self.transition_matrix.nba_2_wgs_1

    @computed_field
    @property
    def carrier_sensitivity(self) -> float:
        """Carrier detection sensitivity."""
        matrix = self.transition_matrix
        true_carriers = matrix.nba_1_wgs_1 + matrix.nba_2_wgs_1 + matrix.nba_1_wgs_2 + matrix.nba_2_wgs_2
        total_carriers = (matrix.nba_0_wgs_1 + matrix.nba_0_wgs_2 +
                         matrix.nba_1_wgs_1 + matrix.nba_1_wgs_2 +
                         matrix.nba_2_wgs_1 + matrix.nba_2_wgs_2)
        return true_carriers / total_carriers if total_carriers > 0 else 0.0

    @computed_field
    @property
    def carrier_specificity(self) -> float:
        """Carrier detection specificity."""
        matrix = self.transition_matrix
        true_non_carriers = matrix.nba_0_wgs_0
        total_non_carriers = matrix.nba_0_wgs_0 + matrix.nba_1_wgs_0 + matrix.nba_2_wgs_0
        return true_non_carriers / total_non_carriers if total_non_carriers > 0 else 0.0

    @computed_field
    @property
    def quality_score(self) -> float:
        """Weighted quality score combining concordance and carrier detection."""
        # Weight overall concordance (70%) + carrier sensitivity (30%)
        return 0.7 * self.overall_concordance + 0.3 * self.carrier_sensitivity


class ProbeAnalysisResult(BaseModel):
    """Analysis results for a single probe using both validation methods."""

    variant_id: str = Field(..., description="NBA probe variant identifier")
    probe_type: Optional[str] = Field(None, description="Probe technology type")
    diagnostic_metrics: DiagnosticMetrics = Field(..., description="Traditional diagnostic test metrics")
    concordance_metrics: ConcordanceMetrics = Field(..., description="Genotype concordance analysis")


class ProbeRecommendation(BaseModel):
    """Probe selection recommendation with rationale."""

    recommended_probe: str = Field(..., description="Selected probe variant_id")
    selection_rationale: str = Field(..., description="Human-readable explanation of selection")
    confidence_score: Optional[float] = Field(None, description="Confidence in recommendation (0-1)")


class MutationAnalysis(BaseModel):
    """Complete analysis for a single mutation with multiple probes."""

    mutation: str = Field(..., description="Mutation name (e.g., p.Leu10Pro)")
    snp_list_id: str = Field(..., description="SNP list identifier")
    chromosome: int = Field(..., description="Chromosome number")
    position: int = Field(..., description="Genomic position")
    wgs_ground_truth_cases: int = Field(..., description="Number of WGS-confirmed carriers")

    probes: List[ProbeAnalysisResult] = Field(..., description="Analysis results for all probes")

    diagnostic_recommendation: ProbeRecommendation = Field(..., description="Recommendation using diagnostic approach")
    concordance_recommendation: ProbeRecommendation = Field(..., description="Recommendation using concordance approach")

    consensus: Dict[str, Any] = Field(..., description="Consensus analysis between methods")


class MethodologyComparison(BaseModel):
    """Comparison between diagnostic and concordance recommendation methods."""

    total_agreements: int = Field(..., description="Number of mutations where methods agree")
    total_disagreements: int = Field(..., description="Number of mutations where methods disagree")
    agreement_rate: float = Field(..., description="Overall agreement rate between methods")
    disagreement_analysis: List[Dict[str, str]] = Field(default_factory=list, description="Details of disagreements")


class ProbeSelectionSummary(BaseModel):
    """High-level summary statistics for the probe selection analysis."""

    total_mutations_analyzed: int = Field(..., description="Total unique mutations examined")
    mutations_with_multiple_probes: int = Field(..., description="Mutations with >1 NBA probe")
    total_probe_comparisons: int = Field(..., description="Total individual probe validations")
    samples_compared: int = Field(..., description="Number of samples with both NBA and WGS data")


class ProbeSelectionReport(BaseModel):
    """Complete probe selection analysis report."""

    job_id: str = Field(..., description="Pipeline job identifier")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="When analysis was performed")

    methodology: Dict[str, Any] = Field(
        default_factory=lambda: {
            "analysis_methods": ["diagnostic", "concordance"],
            "recommendation_strategy": "consensus"
        },
        description="Analysis methodology configuration"
    )

    summary: ProbeSelectionSummary = Field(..., description="High-level summary statistics")
    probe_comparisons: List[MutationAnalysis] = Field(..., description="Per-mutation analysis results")
    methodology_comparison: MethodologyComparison = Field(..., description="Method comparison analysis")


# Type aliases for recommendation strategies
RecommendationStrategy = Literal["diagnostic", "concordance", "consensus"]