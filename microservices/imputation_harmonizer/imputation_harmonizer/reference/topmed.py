"""TOPMed reference panel loader.

[claude-assisted] Implements TOPMed file parsing. Format is identical to HRC
but IDs are NOT rsIDs (format: "TOPMed_freeze_5?chr1:10,498'01").
Supports gzipped files and chromosome filtering for parallel processing.

TOPMed file format (tab-separated):
#CHROM  POS     ID      REF     ALT     AC      AN      AF
1       10498   TOPMed_freeze_5?chr1:10,498'01  G       A       1       125568  7.96381e-06
"""

from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

from imputation_harmonizer.io_utils import smart_open
from imputation_harmonizer.models import ReferenceVariant
from imputation_harmonizer.reference.base import ReferencePanel
from imputation_harmonizer.utils import make_chrpos_key, normalize_chromosome


class TOPMedPanel(ReferencePanel):
    """TOPMed reference panel loader.

    TOPMed IDs are NOT rsIDs (format like "TOPMed_freeze_5?chr1:10,498'01"),
    so ID-based matching is less useful than position-based matching.

    File format columns (same as HRC):
    0: CHROM - Chromosome
    1: POS - Position
    2: ID - TOPMed ID (not rsID)
    3: REF - Reference allele
    4: ALT - Alternate allele
    5: AC - Allele count
    6: AN - Total alleles
    7: AF - Alternate allele frequency

    The 463 million variant TOPMed file requires efficient streaming
    and chromosome filtering for parallel processing.
    """

    def load(
        self,
        filepath: Path,
        population: str = "ALL",  # Ignored for TOPMed
        verbose: bool = False,
        chromosome: str | None = None,
    ) -> None:
        """Load TOPMed reference panel from file.

        Supports gzipped files (.gz) and chromosome filtering for
        parallel processing where each worker loads only one chromosome.

        Args:
            filepath: Path to TOPMed sites file (may be gzipped)
            population: Ignored for TOPMed (single population)
            verbose: Print progress information
            chromosome: If specified, only load variants from this chromosome.
                Useful for parallel processing (e.g., "1", "22", "X").

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not filepath.exists():
            raise FileNotFoundError(f"TOPMed file not found: {filepath}")

        # Normalize filter chromosome if specified
        filter_chr = normalize_chromosome(chromosome) if chromosome else None

        line_count = 0

        chr_desc = f" (chr {chromosome})" if chromosome else ""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Loading TOPMed reference from {filepath.name}{chr_desc}..."
            )

            with smart_open(filepath) as f:
                for line in f:
                    # Skip header lines starting with #
                    if line.startswith("#"):
                        continue

                    # Parse tab-separated line
                    parts = line.rstrip("\n").split("\t")

                    if len(parts) < 8:
                        continue  # Skip malformed lines

                    chr_val = normalize_chromosome(parts[0])

                    # Skip variants not on the target chromosome (for parallel processing)
                    if filter_chr is not None and chr_val != filter_chr:
                        continue

                    line_count += 1

                    # Progress indicator every 1M lines (TOPMed is huge - 463M variants)
                    if verbose and line_count % 1000000 == 0:
                        progress.update(
                            task,
                            description=f"Loading TOPMed{chr_desc}... {line_count:,} variants",
                        )

                    pos = int(parts[1])
                    snp_id = parts[2]
                    ref = parts[3]
                    alt = parts[4]
                    # parts[5] = AC (allele count) - not used
                    # parts[6] = AN (total alleles) - not used
                    af = float(parts[7])

                    # Create chr-pos key
                    chrpos = make_chrpos_key(chr_val, pos)

                    # Create variant
                    variant = ReferenceVariant(
                        chr=chr_val,
                        pos=pos,
                        id=snp_id,
                        ref=ref,
                        alt=alt,
                        alt_af=af,
                    )

                    # Store in position index
                    self._by_position[chrpos] = variant

                    # TOPMed IDs are not rsIDs, but we still index them
                    # for potential ID-based lookup (though position matching is preferred)
                    # Skip IDs that are just "." (unnamed variants)
                    if snp_id != ".":
                        self._id_to_chrpos[snp_id] = chrpos

        if verbose:
            print(f"Loaded {len(self):,} variants from TOPMed reference")
