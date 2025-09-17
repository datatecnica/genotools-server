"""
Probe recommendation engine for selecting optimal probes from analysis results.

Provides configurable strategies for probe selection based on diagnostic
or concordance metrics, with consensus recommendations when methods agree.
"""

import logging
from typing import Dict, List, Optional, Tuple

from ..models.probe_validation import (
    ProbeAnalysisResult,
    ProbeRecommendation,
    MutationAnalysis,
    MethodologyComparison,
    RecommendationStrategy
)

logger = logging.getLogger(__name__)


class ProbeRecommendationEngine:
    """
    Engine for making probe selection recommendations based on analysis results.

    Supports multiple recommendation strategies and provides consensus analysis
    when using multiple approaches.
    """

    def __init__(
        self,
        strategy: RecommendationStrategy = "consensus",
        sensitivity_threshold: float = 0.80,
        specificity_threshold: float = 0.95,
        concordance_threshold: float = 0.90
    ):
        """
        Initialize recommendation engine with strategy and thresholds.

        Args:
            strategy: Recommendation strategy to use
            sensitivity_threshold: Minimum acceptable sensitivity
            specificity_threshold: Minimum acceptable specificity
            concordance_threshold: Minimum acceptable concordance
        """
        self.strategy = strategy
        self.sensitivity_threshold = sensitivity_threshold
        self.specificity_threshold = specificity_threshold
        self.concordance_threshold = concordance_threshold

    def recommend_probes(
        self,
        probe_analysis_by_mutation: Dict[str, List[ProbeAnalysisResult]],
        mutation_metadata: Dict[str, Dict]
    ) -> Tuple[List[MutationAnalysis], MethodologyComparison]:
        """
        Generate probe recommendations for all mutations.

        Args:
            probe_analysis_by_mutation: Analysis results grouped by mutation
            mutation_metadata: Additional metadata for each mutation

        Returns:
            Tuple of (mutation_analyses, methodology_comparison)
        """
        logger.info(f"Generating probe recommendations using {self.strategy} strategy")

        mutation_analyses = []
        diagnostic_choices = []
        concordance_choices = []

        for mutation, probe_results in probe_analysis_by_mutation.items():
            if len(probe_results) < 2:
                # Skip mutations with only one probe
                continue

            logger.debug(f"Analyzing {len(probe_results)} probes for mutation: {mutation}")

            # Get recommendations from both approaches
            diagnostic_rec = self._diagnostic_recommendation(probe_results)
            concordance_rec = self._concordance_recommendation(probe_results)

            # Determine consensus
            consensus = self._determine_consensus(diagnostic_rec, concordance_rec)

            # Build mutation analysis
            metadata = mutation_metadata.get(mutation, {})
            mutation_analysis = MutationAnalysis(
                mutation=mutation,
                snp_list_id=metadata.get('snp_list_id', mutation),
                chromosome=metadata.get('chromosome', 0),
                position=metadata.get('position', 0),
                wgs_ground_truth_cases=metadata.get('wgs_cases', 0),
                probes=probe_results,
                diagnostic_recommendation=diagnostic_rec,
                concordance_recommendation=concordance_rec,
                consensus=consensus
            )

            mutation_analyses.append(mutation_analysis)
            diagnostic_choices.append(diagnostic_rec.recommended_probe)
            concordance_choices.append(concordance_rec.recommended_probe)

        # Generate methodology comparison
        methodology_comparison = self._compare_methodologies(
            mutation_analyses, diagnostic_choices, concordance_choices
        )

        logger.info(f"Generated recommendations for {len(mutation_analyses)} mutations")
        logger.info(f"Method agreement rate: {methodology_comparison.agreement_rate:.3f}")

        return mutation_analyses, methodology_comparison

    def _diagnostic_recommendation(self, probe_results: List[ProbeAnalysisResult]) -> ProbeRecommendation:
        """
        Select probe based on diagnostic metrics (sensitivity priority).

        Args:
            probe_results: List of probe analysis results

        Returns:
            ProbeRecommendation based on diagnostic metrics
        """
        # Sort by sensitivity (desc), then specificity (desc)
        sorted_probes = sorted(
            probe_results,
            key=lambda p: (p.diagnostic_metrics.sensitivity, p.diagnostic_metrics.specificity),
            reverse=True
        )

        best_probe = sorted_probes[0]
        sensitivity = best_probe.diagnostic_metrics.sensitivity
        specificity = best_probe.diagnostic_metrics.specificity

        # Generate rationale
        if sensitivity >= self.sensitivity_threshold and specificity >= self.specificity_threshold:
            rationale = f"Excellent performance: sensitivity {sensitivity:.3f}, specificity {specificity:.3f}"
            confidence = 0.95
        elif sensitivity >= self.sensitivity_threshold:
            rationale = f"Good sensitivity {sensitivity:.3f}, moderate specificity {specificity:.3f}"
            confidence = 0.80
        else:
            rationale = f"Best available: sensitivity {sensitivity:.3f}, specificity {specificity:.3f}"
            confidence = 0.60

        return ProbeRecommendation(
            recommended_probe=best_probe.variant_id,
            selection_rationale=rationale,
            confidence_score=confidence
        )

    def _concordance_recommendation(self, probe_results: List[ProbeAnalysisResult]) -> ProbeRecommendation:
        """
        Select probe based on concordance metrics (overall accuracy priority).

        Args:
            probe_results: List of probe analysis results

        Returns:
            ProbeRecommendation based on concordance metrics
        """
        # Sort by overall concordance (desc), then carrier sensitivity (desc)
        sorted_probes = sorted(
            probe_results,
            key=lambda p: (p.concordance_metrics.overall_concordance, p.concordance_metrics.carrier_sensitivity),
            reverse=True
        )

        best_probe = sorted_probes[0]
        concordance = best_probe.concordance_metrics.overall_concordance
        carrier_sens = best_probe.concordance_metrics.carrier_sensitivity

        # Generate rationale
        if concordance >= self.concordance_threshold:
            rationale = f"Excellent concordance {concordance:.3f}, carrier sensitivity {carrier_sens:.3f}"
            confidence = 0.95
        elif concordance >= 0.85:
            rationale = f"Good concordance {concordance:.3f}, carrier sensitivity {carrier_sens:.3f}"
            confidence = 0.80
        else:
            rationale = f"Best available: concordance {concordance:.3f}, carrier sensitivity {carrier_sens:.3f}"
            confidence = 0.60

        return ProbeRecommendation(
            recommended_probe=best_probe.variant_id,
            selection_rationale=rationale,
            confidence_score=confidence
        )

    def _determine_consensus(
        self,
        diagnostic_rec: ProbeRecommendation,
        concordance_rec: ProbeRecommendation
    ) -> Dict:
        """
        Determine consensus between diagnostic and concordance recommendations.

        Args:
            diagnostic_rec: Diagnostic-based recommendation
            concordance_rec: Concordance-based recommendation

        Returns:
            Dictionary with consensus analysis
        """
        both_agree = diagnostic_rec.recommended_probe == concordance_rec.recommended_probe

        if both_agree:
            # Calculate combined confidence
            combined_confidence = (
                (diagnostic_rec.confidence_score or 0.5) +
                (concordance_rec.confidence_score or 0.5)
            ) / 2

            return {
                "both_methods_agree": True,
                "recommended_probe": diagnostic_rec.recommended_probe,
                "combined_confidence": combined_confidence,
                "rationale": "Both diagnostic and concordance methods recommend the same probe"
            }
        else:
            return {
                "both_methods_agree": False,
                "recommended_probe": None,
                "diagnostic_choice": diagnostic_rec.recommended_probe,
                "concordance_choice": concordance_rec.recommended_probe,
                "rationale": "Methods disagree - manual review recommended",
                "diagnostic_rationale": diagnostic_rec.selection_rationale,
                "concordance_rationale": concordance_rec.selection_rationale
            }

    def _compare_methodologies(
        self,
        mutation_analyses: List[MutationAnalysis],
        diagnostic_choices: List[str],
        concordance_choices: List[str]
    ) -> MethodologyComparison:
        """
        Compare diagnostic vs concordance methodology performance.

        Args:
            mutation_analyses: List of mutation analyses
            diagnostic_choices: List of diagnostic recommendations
            concordance_choices: List of concordance recommendations

        Returns:
            MethodologyComparison with agreement statistics
        """
        total_comparisons = len(diagnostic_choices)
        agreements = sum(1 for d, c in zip(diagnostic_choices, concordance_choices) if d == c)
        disagreements = total_comparisons - agreements

        # Analyze disagreements
        disagreement_analysis = []
        for i, (mutation_analysis, diag_choice, conc_choice) in enumerate(
            zip(mutation_analyses, diagnostic_choices, concordance_choices)
        ):
            if diag_choice != conc_choice:
                disagreement_analysis.append({
                    "mutation": mutation_analysis.mutation,
                    "diagnostic_choice": diag_choice,
                    "concordance_choice": conc_choice,
                    "reason": self._analyze_disagreement_reason(mutation_analysis, diag_choice, conc_choice)
                })

        return MethodologyComparison(
            total_agreements=agreements,
            total_disagreements=disagreements,
            agreement_rate=agreements / total_comparisons if total_comparisons > 0 else 0.0,
            disagreement_analysis=disagreement_analysis
        )

    def _analyze_disagreement_reason(
        self,
        mutation_analysis: MutationAnalysis,
        diagnostic_choice: str,
        concordance_choice: str
    ) -> str:
        """
        Analyze why diagnostic and concordance methods disagreed.

        Args:
            mutation_analysis: Mutation analysis with probe results
            diagnostic_choice: Probe chosen by diagnostic method
            concordance_choice: Probe chosen by concordance method

        Returns:
            Human-readable explanation of disagreement
        """
        # Find the specific probes
        diag_probe = next(
            (p for p in mutation_analysis.probes if p.variant_id == diagnostic_choice),
            None
        )
        conc_probe = next(
            (p for p in mutation_analysis.probes if p.variant_id == concordance_choice),
            None
        )

        if not diag_probe or not conc_probe:
            return "Unable to analyze disagreement"

        # Compare key metrics
        diag_sens = diag_probe.diagnostic_metrics.sensitivity
        diag_spec = diag_probe.diagnostic_metrics.specificity
        conc_sens = conc_probe.diagnostic_metrics.sensitivity
        conc_spec = conc_probe.diagnostic_metrics.specificity

        diag_concordance = diag_probe.concordance_metrics.overall_concordance
        conc_concordance = conc_probe.concordance_metrics.overall_concordance

        if diag_sens > conc_sens and diag_concordance < conc_concordance:
            return "Diagnostic prioritized sensitivity, concordance prioritized overall accuracy"
        elif diag_spec > conc_spec and diag_concordance < conc_concordance:
            return "Diagnostic prioritized specificity, concordance prioritized overall accuracy"
        elif abs(diag_sens - conc_sens) < 0.05 and abs(diag_concordance - conc_concordance) < 0.05:
            return "Methods chose different probes with similar performance metrics"
        else:
            return "Complex trade-off between sensitivity, specificity, and concordance"