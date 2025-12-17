# HRC/1000G Pre-Imputation Checking Tool - Python Implementation

## Project Overview

Rewrite the Will Rayner HRC-1000G-check-bim Perl script (v4.2) as a modern, high-performance Python tool. This is a critical QC step used before submitting GWAS data to imputation servers (Michigan, TOPMed).

### Original Script Reference
- Author: W. Rayner (wrayner@well.ox.ac.uk), 2015
- Original: `HRC-1000G-check-bim-v4.2.pl`
- Memory: ~20GB RAM in Perl version
- Purpose: Compare PLINK .bim files against HRC/1000G reference panels for strand, ID, position, and allele QC

### Goals
1. **Performance**: 10-50x faster via vectorized operations
2. **Memory efficiency**: Streaming/chunked processing for large files
3. **Maintainability**: Type hints, dataclasses, comprehensive tests
4. **Usability**: Modern CLI with clear help and sensible defaults

---

## Functional Specification

### What the Tool Does

Compares your GWAS dataset (.bim + .frq files) against a reference panel (HRC or 1000G) and identifies variants that need:
- Position updates
- Chromosome updates
- Strand flips
- Reference allele reassignment
- Exclusion (problematic variants)

### Input Files

| File | Format | Description |
|------|--------|-------------|
| BIM file | PLINK .bim | 6 columns: chr, rsID, genetic_dist, position, allele1, allele2 |
| FRQ file | PLINK .frq | Allele frequencies from `plink --freq` |
| Reference | HRC or 1000G | Reference panel sites with positions, alleles, frequencies |

#### HRC Reference Format (tab-separated)
```
#CHROM  POS     ID      REF     ALT     AC      AN      AF
1       10177   rs367896724     A       AC      2130    64940   0.0328
```

#### 1000G Legend Format (space/tab-separated)
```
id      chr     position        a0      a1      TYPE    AFR     AMR     EAS     EUR     SAS     ALL
rs367896724:10177:A:AC  1       10177   A       AC      Biallelic_INDEL 0.02    0.17    0.00    0.14    0.07    0.08
```

#### PLINK BIM Format (tab/space-separated)
```
1       rs367896724     0       10177   A       C
```

#### PLINK FRQ Format
```
CHR     SNP             A1      A2      MAF     NCHROBS
1       rs367896724     A       C       0.0328  1000
```

### Checks Performed

| Check | Condition | Action |
|-------|-----------|--------|
| **Not in reference** | chr:pos not found, rsID not found | Exclude |
| **Position mismatch** | rsID found but at different position | Update position |
| **Chromosome mismatch** | rsID found but on different chromosome | Update chromosome |
| **ID mismatch** | Same position, different rsID | Update ID (optional) |
| **Strand flip needed** | Alleles are reverse complement | Flip strand |
| **Ref/Alt swap** | Correct alleles, wrong assignment | Force reference allele |
| **Palindromic SNP (MAF>0.4)** | A/T or G/C SNP with ambiguous strand | Exclude |
| **Frequency difference >0.2** | Large AF discrepancy vs reference | Exclude |
| **Indels** | Insertion/deletion variants | Exclude (HRC r1 has no indels) |
| **Duplicates** | Same chr:pos:alleles after corrections | Exclude |
| **Non-standard chromosomes** | X, Y, XY, MT (HRC) or MT (1000G) | Exclude |

### Output Files

All output files use pattern: `{Type}-{bim_stem}-{Panel}.txt`

| File | Content | Used By |
|------|---------|---------|
| `Exclude-{stem}-{panel}.txt` | SNP IDs to remove | `plink --exclude` |
| `Strand-Flip-{stem}-{panel}.txt` | SNP IDs needing strand flip | `plink --flip` |
| `Force-Allele1-{stem}-{panel}.txt` | SNP ID + correct ref allele | `plink --reference-allele` |
| `Position-{stem}-{panel}.txt` | SNP ID + new position | `plink --update-map` |
| `Chromosome-{stem}-{panel}.txt` | SNP ID + new chromosome | `plink --update-map --update-chr` |
| `ID-{stem}-{panel}.txt` | Old ID + new ID | `plink --update-name` |
| `FreqPlot-{stem}-{panel}.txt` | Frequency comparison data | Plotting/QC |
| `LOG-{stem}-{panel}.txt` | Detailed run statistics | Review |
| `Run-plink.sh` | Ready-to-run PLINK commands | Execution |

### Generated Shell Script Structure

```bash
#!/bin/bash
plink --bfile {stem} --exclude Exclude-{stem}-{panel}.txt --make-bed --out TEMP1
plink --bfile TEMP1 --update-map Chromosome-{stem}-{panel}.txt --update-chr --make-bed --out TEMP2
plink --bfile TEMP2 --update-map Position-{stem}-{panel}.txt --make-bed --out TEMP3
plink --bfile TEMP3 --flip Strand-Flip-{stem}-{panel}.txt --make-bed --out TEMP4
plink --bfile TEMP4 --reference-allele Force-Allele1-{stem}-{panel}.txt --make-bed --out {stem}-updated

# Split by chromosome
for i in {1..22}; do
    plink --bfile {stem}-updated --reference-allele Force-Allele1-{stem}-{panel}.txt --chr $i --make-bed --out {stem}-updated-chr$i
done

rm TEMP*
```

---

## Technical Architecture

### Project Structure

```
imputation_harmonizer/
├── __init__.py
├── __main__.py           # Entry point: python -m imputation_harmonizer
├── cli.py                # Typer CLI definition
├── config.py             # Configuration dataclass and defaults
├── models.py             # Data models (Variant, CheckResult, etc.)
├── reference/
│   ├── __init__.py
│   ├── base.py           # Abstract base class for reference panels
│   ├── hrc.py            # HRC-specific loader
│   └── kg.py             # 1000G-specific loader
├── parsers/
│   ├── __init__.py
│   ├── bim.py            # BIM file parser
│   └── frq.py            # FRQ file parser
├── checks/
│   ├── __init__.py
│   ├── position.py       # Position/chromosome matching
│   ├── strand.py         # Strand flip detection
│   ├── allele.py         # Allele matching and frequency checks
│   └── duplicates.py     # Duplicate detection
├── writers/
│   ├── __init__.py
│   ├── plink_files.py    # Update file writers
│   ├── shell_script.py   # Run-plink.sh generator
│   └── log.py            # Statistics and logging
└── utils.py              # Complement function, helpers

tests/
├── __init__.py
├── conftest.py           # Pytest fixtures
├── test_reference.py
├── test_parsers.py
├── test_checks.py
├── test_integration.py
└── fixtures/
    ├── sample.bim
    ├── sample.frq
    ├── mini_hrc.tab
    └── mini_1000g.legend
```

### Core Data Models

```python
# models.py
from dataclasses import dataclass, field
from typing import Literal, Optional
from enum import Enum, auto

class ExcludeReason(Enum):
    NOT_IN_REFERENCE = auto()
    INDEL = auto()
    PALINDROMIC_HIGH_MAF = auto()
    FREQ_DIFF_TOO_HIGH = auto()
    ALLELE_MISMATCH = auto()
    DUPLICATE = auto()
    ALT_CHROMOSOME = auto()  # X, Y, XY, MT for HRC

class StrandAction(Enum):
    NONE = auto()
    FLIP = auto()

class AlleleAction(Enum):
    NONE = auto()
    FORCE_REF = auto()

@dataclass(slots=True)
class ReferenceVariant:
    """Variant from reference panel (HRC or 1000G)"""
    chr: str
    pos: int
    id: str
    ref: str
    alt: str
    alt_af: float  # Alternate allele frequency

@dataclass(slots=True)
class BimVariant:
    """Variant from PLINK .bim file"""
    chr: str
    id: str
    genetic_dist: float
    pos: int
    allele1: str
    allele2: str
    freq: Optional[float] = None  # Populated from .frq file

@dataclass
class CheckResult:
    """Result of checking a single variant"""
    snp_id: str
    
    # Match info
    matched_by: Literal["position", "id", "none"]
    ref_variant: Optional[ReferenceVariant] = None
    
    # Actions
    exclude: bool = False
    exclude_reason: Optional[ExcludeReason] = None
    
    strand_action: StrandAction = StrandAction.NONE
    allele_action: AlleleAction = AlleleAction.NONE
    force_ref_allele: Optional[str] = None
    
    # Position/ID updates
    update_position: Optional[int] = None
    update_chromosome: Optional[str] = None
    update_id: Optional[str] = None
    
    # For frequency plot
    ref_freq: Optional[float] = None
    bim_freq: Optional[float] = None
    freq_diff: Optional[float] = None
    check_code: Optional[int] = None  # 1-6 matching original script

@dataclass
class Statistics:
    """Running statistics for the check"""
    total: int = 0
    indels: int = 0
    alt_chr_skipped: int = 0
    
    position_match_id_match: int = 0
    position_match_id_mismatch: int = 0
    id_match_position_mismatch: int = 0
    no_match: int = 0
    
    strand_ok: int = 0
    strand_flip: int = 0
    ref_alt_ok: int = 0
    ref_alt_swap: int = 0
    
    palindromic_excluded: int = 0
    freq_diff_excluded: int = 0
    allele_mismatch: int = 0
    duplicates: int = 0
```

### Configuration

```python
# config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

@dataclass
class Config:
    """Configuration for HRC/1000G check"""
    bim_file: Path
    frq_file: Path
    ref_file: Path
    panel: Literal["hrc", "1000g"]
    
    # Optional settings
    population: str = "ALL"  # For 1000G: AFR, AMR, EAS, EUR, SAS, ALL
    output_dir: Optional[Path] = None
    
    # Thresholds
    freq_diff_threshold: float = 0.2
    palindrome_maf_threshold: float = 0.4
    
    # Behavior
    verbose: bool = False
    keep_indels: bool = False  # Future: check indels
    update_ids: bool = False   # Whether to generate ID update file
    
    # Chromosomes to process
    chromosomes: set[str] = field(default_factory=lambda: {str(i) for i in range(1, 23)})
    include_x: bool = False  # 1000G has X
    
    @property
    def file_stem(self) -> str:
        return self.bim_file.stem
    
    @property
    def panel_name(self) -> str:
        return "HRC" if self.panel == "hrc" else "1000G"
```

### Reference Panel Loading

```python
# reference/base.py
from abc import ABC, abstractmethod
from typing import Optional
from ..models import ReferenceVariant

class ReferencePanel(ABC):
    """Abstract base class for reference panels"""
    
    def __init__(self):
        # chr:pos -> ReferenceVariant
        self._by_position: dict[str, ReferenceVariant] = {}
        # rsID -> chr:pos (for ID-based lookup)
        self._id_to_chrpos: dict[str, str] = {}
    
    @abstractmethod
    def load(self, filepath: Path, population: str = "ALL") -> None:
        """Load reference panel from file"""
        pass
    
    def get_by_position(self, chr: str, pos: int) -> Optional[ReferenceVariant]:
        """Lookup variant by chromosome:position"""
        key = f"{chr}-{pos}"
        return self._by_position.get(key)
    
    def get_by_id(self, snp_id: str) -> Optional[ReferenceVariant]:
        """Lookup variant by rsID"""
        chrpos = self._id_to_chrpos.get(snp_id)
        if chrpos:
            return self._by_position.get(chrpos)
        return None
    
    def get_chrpos_for_id(self, snp_id: str) -> Optional[str]:
        """Get chr-pos string for an rsID"""
        return self._id_to_chrpos.get(snp_id)
    
    def __len__(self) -> int:
        return len(self._by_position)
```

```python
# reference/hrc.py
from pathlib import Path
from .base import ReferencePanel
from ..models import ReferenceVariant

class HRCPanel(ReferencePanel):
    """HRC reference panel loader"""
    
    def load(self, filepath: Path, population: str = "ALL") -> None:
        """
        Load HRC sites file.
        
        Format: #CHROM POS ID REF ALT AC AN AF
        """
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith('#'):
                    continue
                
                # Progress indicator every 100k lines
                if line_num % 100000 == 0:
                    print(f" {line_num}", end="", flush=True)
                
                parts = line.strip().split('\t')
                chr_val = parts[0]
                pos = int(parts[1])
                snp_id = parts[2]
                ref = parts[3]
                alt = parts[4]
                af = float(parts[7])  # AF column
                
                chrpos = f"{chr_val}-{pos}"
                
                variant = ReferenceVariant(
                    chr=chr_val,
                    pos=pos,
                    id=snp_id,
                    ref=ref,
                    alt=alt,
                    alt_af=af
                )
                
                self._by_position[chrpos] = variant
                
                # Only index by ID if it's a real rsID
                if snp_id != '.':
                    self._id_to_chrpos[snp_id] = chrpos
        
        print(" Done")
```

```python
# reference/kg.py
from pathlib import Path
from .base import ReferencePanel
from ..models import ReferenceVariant

class KGPanel(ReferencePanel):
    """1000 Genomes reference panel loader"""
    
    POPULATIONS = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS', 'ALL']
    
    def load(self, filepath: Path, population: str = "ALL") -> None:
        """
        Load 1000G legend file.
        
        Format: id chr position a0 a1 TYPE AFR AMR EAS EUR SAS ALL
        """
        with open(filepath, 'r') as f:
            # Parse header to find population column
            header = f.readline().strip().split()
            
            try:
                freq_col = header.index(population)
            except ValueError:
                raise ValueError(
                    f"Population '{population}' not found. "
                    f"Available: {self.POPULATIONS}"
                )
            
            # Check for TYPE column (for multiallelic filtering)
            type_col = header.index('TYPE') if 'TYPE' in header else None
            
            for line_num, line in enumerate(f, 2):
                if line_num % 100000 == 0:
                    print(f" {line_num}", end="", flush=True)
                
                parts = line.strip().split()
                
                snp_id = parts[0]
                chr_val = parts[1]
                pos = int(parts[2])
                ref = parts[3]
                alt = parts[4]
                af = float(parts[freq_col])
                
                # Handle multiallelic sites
                if type_col and parts[type_col].startswith('Multiallelic'):
                    ref, alt = 'N', 'N'  # Will fail allele check
                
                chrpos = f"{chr_val}-{pos}"
                
                variant = ReferenceVariant(
                    chr=chr_val,
                    pos=pos,
                    id=snp_id,
                    ref=ref,
                    alt=alt,
                    alt_af=af
                )
                
                self._by_position[chrpos] = variant
                
                # Extract rsID from compound ID (e.g., "rs123:10177:A:C")
                if snp_id.startswith('rs'):
                    rs_id = snp_id.split(':')[0]
                    self._id_to_chrpos[rs_id] = chrpos
        
        print(" Done")
```

### Strand and Allele Checking

```python
# checks/strand.py
from typing import Tuple
from ..models import StrandAction, AlleleAction, ExcludeReason

# Complement lookup table
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

def complement(allele: str) -> str:
    """Get complement of an allele"""
    return COMPLEMENT.get(allele, allele)

def complement_pair(a1: str, a2: str) -> Tuple[str, str]:
    """Get complement of an allele pair"""
    return complement(a1), complement(a2)

def is_palindromic(a1: str, a2: str) -> bool:
    """Check if SNP is palindromic (A/T or G/C)"""
    return (a1, a2) in [('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')]

def check_strand_and_alleles(
    ref_alleles: Tuple[str, str],  # (ref, alt) from reference
    bim_alleles: Tuple[str, str],  # (a1, a2) from bim file
    ref_alt_af: float,             # Alt allele frequency in reference
    bim_af: float,                 # Allele frequency in bim file
    palindrome_maf_threshold: float = 0.4,
    freq_diff_threshold: float = 0.2
) -> dict:
    """
    Check strand orientation and allele assignment.
    
    Returns dict with:
        - action: 'keep', 'flip', 'exclude'
        - allele_action: 'none', 'force_ref'
        - exclude_reason: if excluded
        - force_ref_allele: if forcing reference
        - ref_freq: reference frequency for plotting
        - freq_diff: frequency difference
        - check_code: 1-6 matching original script
    """
    ref_a, alt_a = ref_alleles
    bim_a1, bim_a2 = bim_alleles
    
    # Calculate MAF for palindromic check
    maf = min(ref_alt_af, 1 - ref_alt_af)
    
    # Check palindromic SNPs first (absolute exclusion if MAF > threshold)
    if is_palindromic(ref_a, alt_a) and maf > palindrome_maf_threshold:
        return {
            'exclude': True,
            'exclude_reason': ExcludeReason.PALINDROMIC_HIGH_MAF,
            'strand_action': StrandAction.NONE,
            'allele_action': AlleleAction.NONE,
            'check_code': 5
        }
    
    # Get complemented bim alleles
    bim_c1, bim_c2 = complement_pair(bim_a1, bim_a2)
    
    result = {
        'exclude': False,
        'exclude_reason': None,
        'strand_action': StrandAction.NONE,
        'allele_action': AlleleAction.NONE,
        'force_ref_allele': None,
        'ref_freq': None,
        'freq_diff': None,
        'check_code': 0
    }
    
    # Case 1: Strand OK, ref/alt OK
    if ref_a == bim_a1 and alt_a == bim_a2:
        ref_freq = 1 - ref_alt_af
        result.update({
            'strand_action': StrandAction.NONE,
            'allele_action': AlleleAction.NONE,
            'ref_freq': ref_freq,
            'freq_diff': ref_freq - bim_af,
            'check_code': 1
        })
    
    # Case 2: Strand OK, ref/alt swapped
    elif ref_a == bim_a2 and alt_a == bim_a1:
        result.update({
            'strand_action': StrandAction.NONE,
            'allele_action': AlleleAction.FORCE_REF,
            'force_ref_allele': ref_a,
            'ref_freq': ref_alt_af,
            'freq_diff': ref_alt_af - bim_af,
            'check_code': 2
        })
    
    # Case 3: Strand flipped, ref/alt OK
    elif ref_a == bim_c1 and alt_a == bim_c2:
        ref_freq = 1 - ref_alt_af
        result.update({
            'strand_action': StrandAction.FLIP,
            'allele_action': AlleleAction.NONE,
            'ref_freq': ref_freq,
            'freq_diff': ref_freq - bim_af,
            'check_code': 3
        })
    
    # Case 4: Strand flipped, ref/alt swapped
    elif ref_a == bim_c2 and alt_a == bim_c1:
        result.update({
            'strand_action': StrandAction.FLIP,
            'allele_action': AlleleAction.FORCE_REF,
            'force_ref_allele': ref_a,
            'ref_freq': ref_alt_af,
            'freq_diff': ref_alt_af - bim_af,
            'check_code': 4
        })
    
    # No match - allele mismatch
    else:
        return {
            'exclude': True,
            'exclude_reason': ExcludeReason.ALLELE_MISMATCH,
            'strand_action': StrandAction.NONE,
            'allele_action': AlleleAction.NONE,
            'check_code': 0
        }
    
    # Check frequency difference (applies to all non-excluded cases)
    if result['freq_diff'] is not None:
        freq_diff = abs(result['freq_diff'])
        if freq_diff > freq_diff_threshold:
            return {
                'exclude': True,
                'exclude_reason': ExcludeReason.FREQ_DIFF_TOO_HIGH,
                'strand_action': StrandAction.NONE,
                'allele_action': AlleleAction.NONE,
                'ref_freq': result['ref_freq'],
                'freq_diff': result['freq_diff'],
                'check_code': 6
            }
    
    return result
```

### Main Comparison Logic

```python
# checks/comparator.py
from pathlib import Path
from typing import Iterator, Set
from ..models import BimVariant, CheckResult, Statistics, ExcludeReason
from ..reference.base import ReferencePanel
from ..config import Config
from .strand import check_strand_and_alleles

def is_indel(a1: str, a2: str) -> bool:
    """Check if variant is an indel"""
    indel_markers = {'-', 'I', 'D'}
    return a1 in indel_markers or a2 in indel_markers or len(a1) > 1 or len(a2) > 1

def check_variants(
    bim_variants: Iterator[BimVariant],
    reference: ReferencePanel,
    config: Config
) -> Iterator[CheckResult]:
    """
    Main comparison loop - check each BIM variant against reference.
    
    Yields CheckResult for each variant.
    """
    seen: Set[str] = set()  # For duplicate detection: "chr-pos-sorted_alleles"
    stats = Statistics()
    
    valid_chromosomes = config.chromosomes.copy()
    if config.include_x:
        valid_chromosomes.add('23')
        valid_chromosomes.add('X')
    
    for variant in bim_variants:
        stats.total += 1
        chrpos = f"{variant.chr}-{variant.pos}"
        
        # Skip non-standard chromosomes
        if variant.chr not in valid_chromosomes:
            stats.alt_chr_skipped += 1
            yield CheckResult(
                snp_id=variant.id,
                matched_by="none",
                exclude=True,
                exclude_reason=ExcludeReason.ALT_CHROMOSOME
            )
            continue
        
        # Check for indels
        if is_indel(variant.allele1, variant.allele2):
            stats.indels += 1
            yield CheckResult(
                snp_id=variant.id,
                matched_by="none",
                exclude=True,
                exclude_reason=ExcludeReason.INDEL
            )
            continue
        
        # Create sorted allele key for duplicate detection
        sorted_alleles = tuple(sorted([variant.allele1, variant.allele2]))
        allele_key = f"{chrpos}-{sorted_alleles[0]}:{sorted_alleles[1]}"
        
        # Also create complement key for duplicate detection
        from .strand import complement_pair
        comp_alleles = complement_pair(variant.allele1, variant.allele2)
        sorted_comp = tuple(sorted(comp_alleles))
        comp_key = f"{chrpos}-{sorted_comp[0]}:{sorted_comp[1]}"
        
        # Try to find variant in reference
        ref_var = reference.get_by_position(variant.chr, variant.pos)
        matched_by = "position" if ref_var else None
        
        # If not found by position, try by ID
        if not ref_var:
            ref_var = reference.get_by_id(variant.id)
            if ref_var:
                matched_by = "id"
        
        # Not in reference at all
        if not ref_var:
            stats.no_match += 1
            yield CheckResult(
                snp_id=variant.id,
                matched_by="none",
                exclude=True,
                exclude_reason=ExcludeReason.NOT_IN_REFERENCE
            )
            continue
        
        # Check for duplicates
        ref_chrpos = f"{ref_var.chr}-{ref_var.pos}"
        dup_key = f"{ref_chrpos}-{sorted_alleles[0]}:{sorted_alleles[1]}"
        dup_comp_key = f"{ref_chrpos}-{sorted_comp[0]}:{sorted_comp[1]}"
        
        if dup_key in seen or dup_comp_key in seen:
            stats.duplicates += 1
            yield CheckResult(
                snp_id=variant.id,
                matched_by=matched_by,
                ref_variant=ref_var,
                exclude=True,
                exclude_reason=ExcludeReason.DUPLICATE
            )
            continue
        
        seen.add(dup_key)
        seen.add(dup_comp_key)
        
        # Build result
        result = CheckResult(
            snp_id=variant.id,
            matched_by=matched_by,
            ref_variant=ref_var
        )
        
        # Check if position/chromosome needs updating
        if matched_by == "id":
            stats.id_match_position_mismatch += 1
            result.update_position = ref_var.pos
            if variant.chr != ref_var.chr:
                result.update_chromosome = ref_var.chr
        elif matched_by == "position":
            if variant.id == ref_var.id:
                stats.position_match_id_match += 1
            else:
                stats.position_match_id_mismatch += 1
                result.update_id = ref_var.id
        
        # Check strand and alleles
        strand_result = check_strand_and_alleles(
            ref_alleles=(ref_var.ref, ref_var.alt),
            bim_alleles=(variant.allele1, variant.allele2),
            ref_alt_af=ref_var.alt_af,
            bim_af=variant.freq or 0.0,
            palindrome_maf_threshold=config.palindrome_maf_threshold,
            freq_diff_threshold=config.freq_diff_threshold
        )
        
        # Update result with strand check results
        result.exclude = strand_result['exclude']
        result.exclude_reason = strand_result.get('exclude_reason')
        result.strand_action = strand_result['strand_action']
        result.allele_action = strand_result['allele_action']
        result.force_ref_allele = strand_result.get('force_ref_allele')
        result.ref_freq = strand_result.get('ref_freq')
        result.bim_freq = variant.freq
        result.freq_diff = strand_result.get('freq_diff')
        result.check_code = strand_result.get('check_code')
        
        # Update statistics
        if strand_result['exclude']:
            reason = strand_result['exclude_reason']
            if reason == ExcludeReason.PALINDROMIC_HIGH_MAF:
                stats.palindromic_excluded += 1
            elif reason == ExcludeReason.FREQ_DIFF_TOO_HIGH:
                stats.freq_diff_excluded += 1
            elif reason == ExcludeReason.ALLELE_MISMATCH:
                stats.allele_mismatch += 1
        else:
            if strand_result['strand_action'] == StrandAction.NONE:
                stats.strand_ok += 1
            else:
                stats.strand_flip += 1
            
            if strand_result['allele_action'] == AlleleAction.NONE:
                stats.ref_alt_ok += 1
            else:
                stats.ref_alt_swap += 1
        
        yield result
```

### CLI Interface

```python
# cli.py
import typer
from pathlib import Path
from typing import Optional
from enum import Enum

app = typer.Typer(
    name="imputation-harmonizer",
    help="Check PLINK files against HRC/1000G reference panels for pre-imputation QC"
)

class Panel(str, Enum):
    hrc = "hrc"
    kg = "1000g"

class Population(str, Enum):
    AFR = "AFR"
    AMR = "AMR"
    EAS = "EAS"
    EUR = "EUR"
    SAS = "SAS"
    ALL = "ALL"

@app.command()
def check(
    bim: Path = typer.Option(..., "--bim", "-b", help="PLINK .bim file"),
    freq: Path = typer.Option(..., "--freq", "-f", help="PLINK .frq frequency file"),
    ref: Path = typer.Option(..., "--ref", "-r", help="Reference panel file (HRC or 1000G)"),
    panel: Panel = typer.Option(..., "--panel", "-p", help="Reference panel type"),
    population: Population = typer.Option(
        Population.ALL, "--pop", help="1000G population (ignored for HRC)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory (default: current)"
    ),
    freq_diff: float = typer.Option(
        0.2, "--freq-diff", help="Max allele frequency difference threshold"
    ),
    palindrome_maf: float = typer.Option(
        0.4, "--palindrome-maf", help="MAF threshold for excluding palindromic SNPs"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    include_x: bool = typer.Option(
        False, "--include-x", help="Include X chromosome (1000G only)"
    )
):
    """
    Check PLINK .bim file against HRC or 1000G reference panel.
    
    Generates update files for PLINK and a shell script to apply corrections.
    
    Example:
        imputation-harmonizer -b data.bim -f data.frq -r HRC.r1.sites.tab -p hrc
        imputation-harmonizer -b data.bim -f data.frq -r 1000G.legend -p 1000g --pop EUR
    """
    from .config import Config
    from .main import run_check
    
    config = Config(
        bim_file=bim,
        frq_file=freq,
        ref_file=ref,
        panel=panel.value,
        population=population.value,
        output_dir=output_dir,
        freq_diff_threshold=freq_diff,
        palindrome_maf_threshold=palindrome_maf,
        verbose=verbose,
        include_x=include_x
    )
    
    run_check(config)

def main():
    app()

if __name__ == "__main__":
    main()
```

### Output Writers

```python
# writers/plink_files.py
from pathlib import Path
from typing import TextIO
from ..models import CheckResult, StrandAction, AlleleAction

class PlinkFileWriter:
    """Manages all PLINK update file outputs"""
    
    def __init__(self, output_dir: Path, file_stem: str, panel_name: str):
        self.output_dir = output_dir
        self.file_stem = file_stem
        self.panel_name = panel_name
        
        # Open all output files
        self.exclude_file = self._open(f"Exclude-{file_stem}-{panel_name}.txt")
        self.strand_file = self._open(f"Strand-Flip-{file_stem}-{panel_name}.txt")
        self.force_file = self._open(f"Force-Allele1-{file_stem}-{panel_name}.txt")
        self.position_file = self._open(f"Position-{file_stem}-{panel_name}.txt")
        self.chromosome_file = self._open(f"Chromosome-{file_stem}-{panel_name}.txt")
        self.id_file = self._open(f"ID-{file_stem}-{panel_name}.txt")
        self.freq_plot_file = self._open(f"FreqPlot-{file_stem}-{panel_name}.txt")
    
    def _open(self, filename: str) -> TextIO:
        return open(self.output_dir / filename, 'w')
    
    def write_result(self, result: CheckResult) -> None:
        """Write a single check result to appropriate files"""
        
        if result.exclude:
            self.exclude_file.write(f"{result.snp_id}\n")
            return
        
        if result.strand_action == StrandAction.FLIP:
            self.strand_file.write(f"{result.snp_id}\n")
        
        if result.allele_action == AlleleAction.FORCE_REF:
            self.force_file.write(f"{result.snp_id}\t{result.force_ref_allele}\n")
        
        if result.update_position is not None:
            self.position_file.write(f"{result.snp_id}\t{result.update_position}\n")
        
        if result.update_chromosome is not None:
            self.chromosome_file.write(f"{result.snp_id}\t{result.update_chromosome}\n")
        
        if result.update_id is not None:
            self.id_file.write(f"{result.snp_id}\t{result.update_id}\n")
        
        # Write frequency plot data
        if result.ref_freq is not None:
            self.freq_plot_file.write(
                f"{result.snp_id}\t{result.ref_freq}\t{result.bim_freq}\t"
                f"{result.freq_diff}\t{result.check_code}\n"
            )
    
    def close(self) -> None:
        """Close all file handles"""
        for f in [self.exclude_file, self.strand_file, self.force_file,
                  self.position_file, self.chromosome_file, self.id_file,
                  self.freq_plot_file]:
            f.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
```

```python
# writers/shell_script.py
from pathlib import Path

def write_shell_script(
    output_dir: Path,
    file_stem: str,
    panel_name: str,
    plink_path: str = "plink"
) -> None:
    """Generate Run-plink.sh script"""
    
    script_path = output_dir / "Run-plink.sh"
    
    exclude = f"Exclude-{file_stem}-{panel_name}.txt"
    chromosome = f"Chromosome-{file_stem}-{panel_name}.txt"
    position = f"Position-{file_stem}-{panel_name}.txt"
    strand = f"Strand-Flip-{file_stem}-{panel_name}.txt"
    force = f"Force-Allele1-{file_stem}-{panel_name}.txt"
    
    updated = f"{file_stem}-updated"
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# Generated by imputation-harmonizer for {panel_name}\n\n")
        
        # Step 1: Remove excluded SNPs
        f.write(f"{plink_path} --bfile {file_stem} --exclude {exclude} "
                f"--make-bed --out TEMP1\n")
        
        # Step 2: Update chromosomes
        f.write(f"{plink_path} --bfile TEMP1 --update-map {chromosome} "
                f"--update-chr --make-bed --out TEMP2\n")
        
        # Step 3: Update positions
        f.write(f"{plink_path} --bfile TEMP2 --update-map {position} "
                f"--make-bed --out TEMP3\n")
        
        # Step 4: Flip strand
        f.write(f"{plink_path} --bfile TEMP3 --flip {strand} "
                f"--make-bed --out TEMP4\n")
        
        # Step 5: Force reference allele
        f.write(f"{plink_path} --bfile TEMP4 --reference-allele {force} "
                f"--make-bed --out {updated}\n\n")
        
        # Split by chromosome
        f.write("# Split into per-chromosome files\n")
        f.write("for i in {1..22}; do\n")
        f.write(f"    {plink_path} --bfile {updated} --reference-allele {force} "
                f"--chr $i --make-bed --out {updated}-chr$i\n")
        f.write("done\n\n")
        
        # Cleanup
        f.write("rm TEMP*\n")
    
    # Make executable
    script_path.chmod(0o755)
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_strand.py
import pytest
from imputation_harmonizer.checks.strand import (
    complement, complement_pair, is_palindromic, check_strand_and_alleles
)
from imputation_harmonizer.models import StrandAction, AlleleAction, ExcludeReason

class TestComplement:
    def test_complement_bases(self):
        assert complement('A') == 'T'
        assert complement('T') == 'A'
        assert complement('C') == 'G'
        assert complement('G') == 'C'
    
    def test_complement_pair(self):
        assert complement_pair('A', 'C') == ('T', 'G')

class TestPalindromic:
    def test_is_palindromic(self):
        assert is_palindromic('A', 'T') is True
        assert is_palindromic('G', 'C') is True
        assert is_palindromic('A', 'C') is False

class TestStrandCheck:
    def test_strand_ok_ref_alt_ok(self):
        """Case 1: Everything matches"""
        result = check_strand_and_alleles(
            ref_alleles=('A', 'G'),
            bim_alleles=('A', 'G'),
            ref_alt_af=0.3,
            bim_af=0.28
        )
        assert result['exclude'] is False
        assert result['strand_action'] == StrandAction.NONE
        assert result['allele_action'] == AlleleAction.NONE
        assert result['check_code'] == 1
    
    def test_strand_ok_ref_alt_swapped(self):
        """Case 2: Alleles swapped"""
        result = check_strand_and_alleles(
            ref_alleles=('A', 'G'),
            bim_alleles=('G', 'A'),
            ref_alt_af=0.3,
            bim_af=0.28
        )
        assert result['exclude'] is False
        assert result['strand_action'] == StrandAction.NONE
        assert result['allele_action'] == AlleleAction.FORCE_REF
        assert result['force_ref_allele'] == 'A'
        assert result['check_code'] == 2
    
    def test_strand_flip_needed(self):
        """Case 3: Need strand flip"""
        result = check_strand_and_alleles(
            ref_alleles=('A', 'G'),
            bim_alleles=('T', 'C'),
            ref_alt_af=0.3,
            bim_af=0.28
        )
        assert result['exclude'] is False
        assert result['strand_action'] == StrandAction.FLIP
        assert result['check_code'] == 3
    
    def test_palindromic_high_maf_excluded(self):
        """Case 5: Palindromic with MAF > 0.4"""
        result = check_strand_and_alleles(
            ref_alleles=('A', 'T'),
            bim_alleles=('A', 'T'),
            ref_alt_af=0.45,
            bim_af=0.44
        )
        assert result['exclude'] is True
        assert result['exclude_reason'] == ExcludeReason.PALINDROMIC_HIGH_MAF
        assert result['check_code'] == 5
    
    def test_freq_diff_excluded(self):
        """Case 6: Frequency difference > 0.2"""
        result = check_strand_and_alleles(
            ref_alleles=('A', 'G'),
            bim_alleles=('A', 'G'),
            ref_alt_af=0.3,
            bim_af=0.05  # Large difference
        )
        assert result['exclude'] is True
        assert result['exclude_reason'] == ExcludeReason.FREQ_DIFF_TOO_HIGH
        assert result['check_code'] == 6
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
from pathlib import Path
from imputation_harmonizer.main import run_check
from imputation_harmonizer.config import Config

@pytest.fixture
def sample_files(tmp_path):
    """Create sample BIM, FRQ, and reference files"""
    
    # Create sample BIM
    bim = tmp_path / "test.bim"
    bim.write_text(
        "1\trs123\t0\t10000\tA\tG\n"
        "1\trs456\t0\t20000\tC\tT\n"
        "1\trs789\t0\t30000\tA\tT\n"  # Palindromic
    )
    
    # Create sample FRQ
    frq = tmp_path / "test.frq"
    frq.write_text(
        "CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n"
        "1\trs123\tA\tG\t0.30\t1000\n"
        "1\trs456\tC\tT\t0.25\t1000\n"
        "1\trs789\tA\tT\t0.45\t1000\n"
    )
    
    # Create sample HRC reference
    ref = tmp_path / "ref.tab"
    ref.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
        "1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
        "1\t20000\trs456\tC\tT\t250\t1000\t0.25\n"
        "1\t30000\trs789\tA\tT\t450\t1000\t0.45\n"
    )
    
    return {'bim': bim, 'frq': frq, 'ref': ref, 'dir': tmp_path}

def test_full_pipeline(sample_files):
    """Test complete checking pipeline"""
    config = Config(
        bim_file=sample_files['bim'],
        frq_file=sample_files['frq'],
        ref_file=sample_files['ref'],
        panel="hrc",
        output_dir=sample_files['dir']
    )
    
    run_check(config)
    
    # Check output files were created
    assert (sample_files['dir'] / "Exclude-test-HRC.txt").exists()
    assert (sample_files['dir'] / "Run-plink.sh").exists()
    
    # Check palindromic SNP was excluded
    excludes = (sample_files['dir'] / "Exclude-test-HRC.txt").read_text()
    assert "rs789" in excludes
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def mini_hrc(fixtures_dir):
    return fixtures_dir / "mini_hrc.tab"

@pytest.fixture
def mini_1000g(fixtures_dir):
    return fixtures_dir / "mini_1000g.legend"
```

---

## Performance Optimization

### Phase 1: Baseline Implementation
- Pure Python with standard library
- Line-by-line file processing
- Dictionary-based lookups

### Phase 2: Polars Migration (if needed)
```python
# Alternative reference loading with Polars
import polars as pl

def load_hrc_polars(filepath: Path) -> pl.DataFrame:
    return pl.read_csv(
        filepath,
        separator='\t',
        comment_char='#',
        has_header=False,
        new_columns=['chr', 'pos', 'id', 'ref', 'alt', 'ac', 'an', 'af']
    )
```

### Phase 3: Parallel Processing (for very large files)
```python
from concurrent.futures import ProcessPoolExecutor
import itertools

def chunk_iterator(iterator, chunk_size=100000):
    """Yield chunks of items from iterator"""
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk
```

### Memory Optimization
- Use `__slots__` on dataclasses
- Stream BIM file instead of loading entirely
- Clear reference dict after processing if memory constrained

---

## Implementation Phases

### Phase 1: Core Structure (Day 1)
- [ ] Set up project structure with pyproject.toml
- [ ] Implement Config dataclass
- [ ] Implement data models (ReferenceVariant, BimVariant, CheckResult)
- [ ] Create abstract ReferencePanel base class
- [ ] Implement HRC panel loader
- [ ] Implement 1000G panel loader
- [ ] Write unit tests for reference loading

### Phase 2: File Parsers (Day 1-2)
- [ ] Implement BIM file parser (streaming)
- [ ] Implement FRQ file parser
- [ ] Merge frequencies into BIM variants
- [ ] Write unit tests for parsers

### Phase 3: Checking Logic (Day 2)
- [ ] Implement complement/strand functions
- [ ] Implement check_strand_and_alleles function
- [ ] Implement main comparator logic
- [ ] Handle position matching
- [ ] Handle ID matching
- [ ] Handle duplicate detection
- [ ] Write comprehensive unit tests

### Phase 4: Output Generation (Day 2-3)
- [ ] Implement PlinkFileWriter class
- [ ] Implement shell script generator
- [ ] Implement log/statistics writer
- [ ] Generate frequency plot data file

### Phase 5: CLI & Integration (Day 3)
- [ ] Implement Typer CLI
- [ ] Wire up all components in main.py
- [ ] Write integration tests
- [ ] Test against real HRC/1000G files

### Phase 6: Polish (Day 3-4)
- [ ] Add progress indicators
- [ ] Improve error messages
- [ ] Add verbose logging mode
- [ ] Performance profiling
- [ ] Documentation

---

## Dependencies

```toml
# pyproject.toml
[project]
name = "imputation-harmonizer"
version = "1.0.0"
description = "Check PLINK files against HRC/1000G for pre-imputation QC"
requires-python = ">=3.10"
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",  # For progress bars and pretty output
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]
fast = [
    "polars>=0.19.0",  # For faster file processing
]

[project.scripts]
imputation-harmonizer = "imputation_harmonizer.cli:main"
```

---

## Usage Examples

```bash
# Basic HRC check
imputation-harmonizer -b mydata.bim -f mydata.frq -r HRC.r1.GRCh37.autosomes.mac5.sites.tab -p hrc

# 1000G with European population
imputation-harmonizer -b mydata.bim -f mydata.frq -r 1000GP_Phase3_combined.legend -p 1000g --pop EUR

# With custom thresholds
imputation-harmonizer -b mydata.bim -f mydata.frq -r ref.tab -p hrc \
    --freq-diff 0.15 \
    --palindrome-maf 0.35 \
    --output-dir ./qc_results \
    --verbose

# After running, execute the generated script:
bash Run-plink.sh
```

---

## Validation

Compare output against original Perl script:
1. Run both tools on same input files
2. Diff exclude lists, strand flip lists, etc.
3. Verify statistics match within tolerance
4. Test edge cases: empty files, all excluded, no changes needed

---

## Notes for Implementation

1. **File encoding**: HRC/1000G files are ASCII, but handle UTF-8 gracefully
2. **Large files**: Reference panels are ~40M variants; optimize memory
3. **Chromosome naming**: Handle both "1" and "chr1" formats
4. **X chromosome**: In 1000G coded as "23", in HRC as "X" or absent
5. **Missing frequencies**: Handle NA/missing MAF values gracefully
6. **Multiallelic sites**: 1000G marks these; exclude or set to N:N

---

## Success Criteria

1. ✅ Produces identical output files to Perl script (within rounding)
2. ✅ Runs 10x+ faster on benchmark dataset
3. ✅ Uses <10GB RAM on full HRC reference
4. ✅ Clear error messages for malformed input
5. ✅ 90%+ test coverage
6. ✅ Type-checked with mypy