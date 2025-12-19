"""Abstract base class for reference panels.

[claude-assisted] Defines the interface for reference panel loaders,
with dict-based O(1) lookup by position or rsID.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from imputation_harmonizer.models import ReferenceVariant
from imputation_harmonizer.utils import make_chrpos_key


class ReferencePanel(ABC):
    """Abstract base class for reference panel loaders.

    Reference panels (HRC, 1000G) are loaded into memory for O(1) lookup.
    Two indexes are maintained:
    - _by_position: chr-pos -> ReferenceVariant
    - _id_to_chrpos: rsID -> chr-pos

    Subclasses implement the load() method for their specific file format.
    """

    def __init__(self) -> None:
        """Initialize empty reference panel."""
        # Primary index: chr-pos -> ReferenceVariant
        self._by_position: dict[str, ReferenceVariant] = {}
        # Secondary index: rsID -> chr-pos (for ID-based lookup)
        self._id_to_chrpos: dict[str, str] = {}

    @abstractmethod
    def load(
        self,
        filepath: Path,
        population: str = "ALL",
        verbose: bool = False,
        chromosome: str | None = None,
    ) -> None:
        """Load reference panel from file.

        Args:
            filepath: Path to reference panel file (may be gzipped)
            population: Population for frequency column (1000G only)
            verbose: Print progress information
            chromosome: If specified, only load variants from this chromosome.
                Useful for parallel processing where each worker loads one chr.

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        pass

    def get_by_position(self, chr_val: str, pos: int) -> ReferenceVariant | None:
        """Look up variant by chromosome and position.

        Args:
            chr_val: Chromosome value
            pos: Base pair position

        Returns:
            ReferenceVariant if found, None otherwise
        """
        key = make_chrpos_key(chr_val, pos)
        return self._by_position.get(key)

    def get_by_id(self, snp_id: str) -> ReferenceVariant | None:
        """Look up variant by rsID.

        Args:
            snp_id: Variant identifier (rsID)

        Returns:
            ReferenceVariant if found, None otherwise
        """
        chrpos = self._id_to_chrpos.get(snp_id)
        if chrpos:
            return self._by_position.get(chrpos)
        return None

    def get_chrpos_for_id(self, snp_id: str) -> str | None:
        """Get chr-pos string for an rsID.

        Used to check if an rsID exists and get its reference position.

        Args:
            snp_id: Variant identifier (rsID)

        Returns:
            chr-pos string if found, None otherwise
        """
        return self._id_to_chrpos.get(snp_id)

    def has_position(self, chr_val: str, pos: int) -> bool:
        """Check if position exists in reference.

        Args:
            chr_val: Chromosome value
            pos: Base pair position

        Returns:
            True if position exists in reference
        """
        key = make_chrpos_key(chr_val, pos)
        return key in self._by_position

    def has_id(self, snp_id: str) -> bool:
        """Check if rsID exists in reference.

        Args:
            snp_id: Variant identifier (rsID)

        Returns:
            True if rsID exists in reference
        """
        return snp_id in self._id_to_chrpos

    def __len__(self) -> int:
        """Return number of variants in reference panel."""
        return len(self._by_position)

    def __contains__(self, key: str) -> bool:
        """Check if chr-pos key exists in reference."""
        return key in self._by_position

    def clear(self) -> None:
        """Clear all loaded data to free memory."""
        self._by_position.clear()
        self._id_to_chrpos.clear()
