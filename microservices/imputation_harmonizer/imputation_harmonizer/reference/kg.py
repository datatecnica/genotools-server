"""1000 Genomes reference panel loader.

[claude-assisted] Implements 1000G legend file parsing matching the original
Perl script's read_kg() function (lines 757-818). Supports gzipped files and
chromosome filtering for parallel processing.

1000G legend file format (space/tab-separated):
id      chr     position        a0      a1      TYPE    AFR     AMR     EAS     EUR     SAS     ALL
rs367896724:10177:A:AC  1       10177   A       AC      Biallelic_INDEL 0.02    0.17    0.00    0.14    0.07    0.08
"""

from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

from imputation_harmonizer.io_utils import smart_open
from imputation_harmonizer.models import ReferenceVariant
from imputation_harmonizer.reference.base import ReferencePanel
from imputation_harmonizer.utils import extract_rsid, make_chrpos_key, normalize_chromosome


class KGPanel(ReferencePanel):
    """1000 Genomes reference panel loader.

    1000G includes autosomes (1-22) and X chromosome (coded as "23" in some files).
    Supports population-specific frequencies: AFR, AMR, EAS, EUR, SAS, ALL.

    File format columns (header determines column positions):
    - id: Variant ID (compound format: rsID:pos:ref:alt)
    - chr: Chromosome
    - position: Base pair position
    - a0: Reference allele
    - a1: Alternate allele
    - TYPE: Variant type (Biallelic_SNP, Biallelic_INDEL, Multiallelic_*)
    - AFR, AMR, EAS, EUR, SAS, ALL: Population frequencies
    """

    POPULATIONS = ["AFR", "AMR", "EAS", "EUR", "SAS", "ALL"]

    def load(
        self,
        filepath: Path,
        population: str = "ALL",
        verbose: bool = False,
        chromosome: str | None = None,
    ) -> None:
        """Load 1000G reference panel from legend file.

        Supports gzipped files (.gz) and chromosome filtering for
        parallel processing where each worker loads only one chromosome.

        Args:
            filepath: Path to 1000G legend file (may be gzipped)
            population: Population for frequency column (AFR, AMR, EAS, EUR, SAS, ALL)
            verbose: Print progress information
            chromosome: If specified, only load variants from this chromosome.
                Useful for parallel processing (e.g., "1", "22", "X").

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If population not found in header
        """
        if not filepath.exists():
            raise FileNotFoundError(f"1000G file not found: {filepath}")

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
                f"Loading 1000G reference from {filepath.name}{chr_desc}..."
            )

            with smart_open(filepath) as f:
                # Parse header to find population column
                header_line = f.readline().rstrip("\n")
                header = header_line.split()

                # Find frequency column for requested population
                try:
                    freq_col = header.index(population)
                except ValueError:
                    raise ValueError(
                        f"Population '{population}' not found in header. "
                        f"Available: {self.POPULATIONS}"
                    )

                # Find TYPE column (for multiallelic filtering)
                type_col: int | None = None
                if "TYPE" in header:
                    type_col = header.index("TYPE")

                # Process data lines
                for line in f:
                    # Parse space/tab-separated line
                    parts = line.rstrip("\n").split()

                    if len(parts) <= freq_col:
                        continue  # Skip malformed lines

                    chr_val = normalize_chromosome(parts[1])

                    # Skip variants not on the target chromosome (for parallel processing)
                    if filter_chr is not None and chr_val != filter_chr:
                        continue

                    line_count += 1

                    # Progress indicator every 100k lines (matches Perl)
                    if verbose and line_count % 100000 == 0:
                        progress.update(
                            task,
                            description=f"Loading 1000G{chr_desc}... {line_count:,} variants",
                        )

                    snp_id = parts[0]
                    pos = int(parts[2])
                    ref = parts[3]
                    alt = parts[4]
                    af = float(parts[freq_col])

                    # Handle multiallelic sites (set alleles to N:N to fail allele check)
                    # Perl: if ($temp[$typecol] =~ /^Multiallelic.*/) { $refalt{$chrpos} = 'N:N'; }
                    if type_col is not None and parts[type_col].startswith("Multiallelic"):
                        ref = "N"
                        alt = "N"

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

                    # Extract rsID from compound ID (e.g., "rs123:10177:A:C" -> "rs123")
                    # Perl: if ($temp[0] =~ /^rs.*/) { @tempids = split(/:/, $temp[0]); $rs{$tempids[0]} = $chrpos; }
                    rsid = extract_rsid(snp_id)
                    if rsid:
                        self._id_to_chrpos[rsid] = chrpos

        if verbose:
            print(
                f"Loaded {len(self):,} variants from 1000G reference "
                f"(population: {population})"
            )
