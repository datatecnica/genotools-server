"""HRC (Haplotype Reference Consortium) panel loader.

[claude-assisted] Implements HRC file parsing matching the original Perl
script's read_hrc() function (lines 732-755). Supports gzipped files and
chromosome filtering for parallel processing.

HRC file format (tab-separated):
#CHROM  POS     ID      REF     ALT     AC      AN      AF
1       10177   rs367896724     A       AC      2130    64940   0.0328
"""

from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

from imputation_harmonizer.io_utils import smart_open
from imputation_harmonizer.models import ReferenceVariant
from imputation_harmonizer.reference.base import ReferencePanel
from imputation_harmonizer.utils import make_chrpos_key, normalize_chromosome


class HRCPanel(ReferencePanel):
    """HRC reference panel loader.

    HRC r1 contains only autosomes (chr 1-22), no X/Y/MT.
    No indels in HRC r1.

    File format columns:
    0: CHROM - Chromosome
    1: POS - Position
    2: ID - rsID (may be '.' for unnamed variants)
    3: REF - Reference allele
    4: ALT - Alternate allele
    5: AC - Allele count
    6: AN - Total alleles
    7: AF - Alternate allele frequency
    """

    def load(
        self,
        filepath: Path,
        population: str = "ALL",  # Ignored for HRC
        verbose: bool = False,
        chromosome: str | None = None,
    ) -> None:
        """Load HRC reference panel from file.

        Supports gzipped files (.gz) and chromosome filtering for
        parallel processing where each worker loads only one chromosome.

        Args:
            filepath: Path to HRC sites file (may be gzipped)
            population: Ignored for HRC (only one population)
            verbose: Print progress information
            chromosome: If specified, only load variants from this chromosome.
                Useful for parallel processing (e.g., "1", "22", "X").

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not filepath.exists():
            raise FileNotFoundError(f"HRC file not found: {filepath}")

        # Normalize filter chromosome if specified
        filter_chr = normalize_chromosome(chromosome) if chromosome else None

        line_count = 0

        chr_desc = f" (chr {chromosome})" if chromosome else ""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Loading HRC reference from {filepath.name}{chr_desc}...")

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

                    # Progress indicator every 100k lines (matches Perl)
                    if verbose and line_count % 100000 == 0:
                        progress.update(
                            task, description=f"Loading HRC{chr_desc}... {line_count:,} variants"
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

                    # Store in ID index (only if it's a real rsID, not '.')
                    # Perl: if ($temp[2] ne '.') { $rs{$temp[2]} = $chrpos; }
                    if snp_id != ".":
                        self._id_to_chrpos[snp_id] = chrpos

        if verbose:
            print(f"Loaded {len(self):,} variants from HRC reference")
