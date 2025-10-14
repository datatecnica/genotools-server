"""
Utility for loading and processing probe selection results.

Provides functionality to load probe selection JSON files and extract
recommended variant IDs for filtering downstream analyses.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class ProbeSelectionLoader:
    """Loads and processes probe selection analysis results."""

    def __init__(self, probe_selection_path: Optional[str] = None):
        """Initialize loader with optional probe selection file path.

        Args:
            probe_selection_path: Path to probe selection JSON file
        """
        self.probe_selection_path = probe_selection_path
        self.selection_map: Dict[str, str] = {}
        self.selected_variant_ids: Set[str] = set()
        self.mutations_analyzed = 0
        self.probes_excluded = 0

        if probe_selection_path:
            self._load_probe_selection()

    def _load_probe_selection(self) -> None:
        """Load probe selection JSON and build variant selection map."""
        if not self.probe_selection_path:
            return

        path = Path(self.probe_selection_path)
        if not path.exists():
            logger.warning(f"Probe selection file not found: {self.probe_selection_path}")
            return

        try:
            logger.info(f"Loading probe selection from: {self.probe_selection_path}")
            with open(path, 'r') as f:
                data = json.load(f)

            probe_comparisons = data.get('probe_comparisons', [])
            self.mutations_analyzed = len(probe_comparisons)

            for comparison in probe_comparisons:
                snp_list_id = comparison.get('snp_list_id')
                consensus = comparison.get('consensus', {})
                probes = comparison.get('probes', [])

                # Get recommended probe (consensus or fallback to diagnostic)
                recommended_probe = consensus.get('recommended_probe')

                # If consensus is None (disagreement), fall back to diagnostic recommendation
                if not recommended_probe:
                    diagnostic_rec = comparison.get('diagnostic_recommendation', {})
                    recommended_probe = diagnostic_rec.get('recommended_probe')
                    if recommended_probe:
                        logger.warning(
                            f"Methods disagree for {snp_list_id}, using diagnostic recommendation: {recommended_probe}"
                        )

                if snp_list_id and recommended_probe:
                    self.selection_map[snp_list_id] = recommended_probe
                    self.selected_variant_ids.add(recommended_probe)

                    # Count excluded probes
                    total_probes = len(probes)
                    if total_probes > 1:
                        self.probes_excluded += (total_probes - 1)

            logger.info(
                f"Loaded probe selection: {len(self.selection_map)} mutations, "
                f"{self.probes_excluded} inferior probes to exclude"
            )

        except Exception as e:
            logger.error(f"Failed to load probe selection: {e}")
            self.selection_map = {}
            self.selected_variant_ids = set()

    def get_recommended_variant(self, snp_list_id: str) -> Optional[str]:
        """Get recommended variant_id for a given snp_list_id.

        Args:
            snp_list_id: SNP list identifier

        Returns:
            Recommended variant_id, or None if not in selection map
        """
        return self.selection_map.get(snp_list_id)

    def is_selected_variant(self, variant_id: str) -> bool:
        """Check if a variant_id is among the selected probes.

        Args:
            variant_id: Variant identifier to check

        Returns:
            True if variant is selected, False otherwise
        """
        return variant_id in self.selected_variant_ids

    def has_probe_selection(self) -> bool:
        """Check if probe selection data was successfully loaded.

        Returns:
            True if probe selection is available, False otherwise
        """
        return len(self.selection_map) > 0

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about probe selection.

        Returns:
            Dictionary with counts of mutations analyzed and probes excluded
        """
        return {
            'mutations_analyzed': self.mutations_analyzed,
            'mutations_with_selection': len(self.selection_map),
            'total_selected_variants': len(self.selected_variant_ids),
            'probes_excluded': self.probes_excluded
        }
