# CLAUDE.md - HRC/1000G Pre-Imputation Check Tool

## Project Scope

Claude is used as an AI coding assistant for this Python CLI tool that performs QC checks on GWAS data before imputation. The tool compares PLINK .bim files against HRC/1000G reference panels to identify strand flips, allele mismatches, position errors, and problematic variants. This is a rewrite of the Will Rayner Perl script (HRC-1000G-check-bim-v4.2.pl) with improved performance and maintainability.

All code must satisfy standards for correctness, reproducibility, and performance with large genomic datasets (40M+ reference variants).

---

## Core Requirements

| Requirement | Description |
|-------------|-------------|
| **Empirical Validation** | All changes must be tested against known input/output from the original Perl script |
| **Human Review** | AI-generated code requires review—genomics logic and edge cases must be verified by domain expert |
| **Incrementalism** | Submit small, atomic changes. Implement one module at a time (reference loading → parsing → checks → output) |
| **Testing** | Use pytest. All checking functions require unit tests with genomic edge cases |
| **Performance** | Must handle 40M+ variants; target <10GB RAM and 10x faster than Perl |

---

## Key Conventions

### Python Standards

- **Python 3.10+** required
- **Lint**: `ruff` (preferred) or `flake8`
- **Format**: `black` (line-length 88)
- **Type-check**: `mypy --strict`
- **Import sort**: `isort` (black profile)

Run all checks before committing:
```bash
black .
isort .
ruff check .  # or flake8
mypy imputation_harmonizer/
pytest tests/ -v
```

### Type Hints

Required for ALL public functions. Use modern syntax:

```python
# ✅ GOOD - Modern Python 3.10+ syntax
def check_variant(
    variant: BimVariant,
    reference: dict[str, ReferenceVariant],
    threshold: float = 0.2,
) -> CheckResult:
    ...

# ✅ GOOD - Use | for Optional
def get_frequency(snp_id: str) -> float | None:
    ...

# ❌ BAD - Old typing module style (avoid unless necessary)
from typing import Dict, Optional
def check_variant(reference: Dict[str, ReferenceVariant]) -> Optional[float]:
    ...
```

### Docstrings

Use Google-style docstrings for all public functions:

```python
def check_strand_and_alleles(
    ref_alleles: tuple[str, str],
    bim_alleles: tuple[str, str],
    ref_alt_af: float,
    bim_af: float,
) -> StrandCheckResult:
    """
    Check strand orientation and allele assignment against reference.

    Compares alleles from BIM file against reference panel, detecting
    strand flips and ref/alt swaps. Excludes palindromic SNPs with
    MAF > 0.4 and variants with frequency difference > 0.2.

    Args:
        ref_alleles: (ref, alt) tuple from reference panel
        bim_alleles: (a1, a2) tuple from BIM file
        ref_alt_af: Alternate allele frequency in reference
        bim_af: Allele frequency from dataset .frq file

    Returns:
        StrandCheckResult with action (keep/flip/exclude) and metadata

    Raises:
        ValueError: If alleles contain invalid characters

    Note:
        [claude-assisted] Implements same logic as check_strand() in
        original Perl script lines 595-723
    """
```

---

## Project Structure

```
imputation_harmonizer/
├── __init__.py
├── __main__.py           # Entry point: python -m imputation_harmonizer
├── cli.py                # Typer CLI definition
├── config.py             # Configuration dataclass
├── models.py             # Data models (dataclasses)
├── reference/
│   ├── __init__.py
│   ├── base.py           # Abstract ReferencePanel class
│   ├── hrc.py            # HRC panel loader
│   └── kg.py             # 1000G panel loader
├── parsers/
│   ├── __init__.py
│   ├── bim.py            # BIM file streaming parser
│   └── frq.py            # FRQ file parser
├── checks/
│   ├── __init__.py
│   ├── strand.py         # Strand/allele checking
│   ├── position.py       # Position matching logic
│   └── duplicates.py     # Duplicate detection
├── writers/
│   ├── __init__.py
│   ├── plink_files.py    # PLINK update file writers
│   ├── shell_script.py   # Run-plink.sh generator
│   └── log.py            # Statistics and logging
└── utils.py              # Complement function, helpers

tests/
├── __init__.py
├── conftest.py           # Pytest fixtures
├── test_reference.py     # Reference loading tests
├── test_parsers.py       # Parser tests
├── test_strand.py        # Strand checking tests
├── test_checks.py        # Integration tests
└── fixtures/
    ├── sample.bim
    ├── sample.frq
    ├── mini_hrc.tab
    └── mini_1000g.legend
```

### Module Responsibilities

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `cli.py` | Typer commands, argument parsing | `check()` command |
| `config.py` | Configuration dataclass | `Config` with all options |
| `models.py` | Data structures | `ReferenceVariant`, `BimVariant`, `CheckResult` |
| `reference/` | Load reference panels | `HRCPanel.load()`, `KGPanel.load()` |
| `parsers/` | Stream input files | `parse_bim()`, `parse_frq()` |
| `checks/` | Comparison logic | `check_strand_and_alleles()`, `is_duplicate()` |
| `writers/` | Generate outputs | `PlinkFileWriter`, `write_shell_script()` |

---

## Data Model Patterns

Use `@dataclass` with `slots=True` for memory efficiency. Do NOT use Pydantic.

```python
from dataclasses import dataclass
from enum import Enum, auto

class ExcludeReason(Enum):
    """Reasons for excluding a variant."""
    NOT_IN_REFERENCE = auto()
    INDEL = auto()
    PALINDROMIC_HIGH_MAF = auto()
    FREQ_DIFF_TOO_HIGH = auto()
    ALLELE_MISMATCH = auto()
    DUPLICATE = auto()
    ALT_CHROMOSOME = auto()

@dataclass(slots=True)
class ReferenceVariant:
    """Variant from HRC or 1000G reference panel."""
    chr: str
    pos: int
    id: str
    ref: str
    alt: str
    alt_af: float

@dataclass(slots=True)
class BimVariant:
    """Variant from PLINK .bim file."""
    chr: str
    id: str
    genetic_dist: float
    pos: int
    allele1: str
    allele2: str
    freq: float | None = None

@dataclass
class CheckResult:
    """Result of checking a single variant."""
    snp_id: str
    exclude: bool = False
    exclude_reason: ExcludeReason | None = None
    flip_strand: bool = False
    force_ref_allele: str | None = None
    update_position: int | None = None
    update_chromosome: str | None = None
```

---

## CLI Pattern (Typer)

This is a CLI tool. Use Typer, NOT FastAPI.

```python
# cli.py
import typer
from pathlib import Path
from typing import Annotated
from enum import Enum

app = typer.Typer(
    name="imputation-harmonizer",
    help="Check PLINK files against HRC/1000G for pre-imputation QC",
    add_completion=False,
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
    bim: Annotated[Path, typer.Option("--bim", "-b", help="PLINK .bim file")],
    freq: Annotated[Path, typer.Option("--freq", "-f", help="PLINK .frq file")],
    ref: Annotated[Path, typer.Option("--ref", "-r", help="Reference panel file")],
    panel: Annotated[Panel, typer.Option("--panel", "-p", help="Panel type")],
    population: Annotated[Population, typer.Option("--pop")] = Population.ALL,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o")] = None,
    freq_diff: Annotated[float, typer.Option("--freq-diff")] = 0.2,
    palindrome_maf: Annotated[float, typer.Option("--palindrome-maf")] = 0.4,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """
    Check PLINK .bim file against HRC or 1000G reference panel.

    Generates PLINK update files and a shell script to apply corrections.
    """
    from .config import Config
    from .main import run_check

    config = Config(
        bim_file=bim,
        frq_file=freq,
        ref_file=ref,
        panel=panel.value,
        population=population.value,
        output_dir=output_dir or Path.cwd(),
        freq_diff_threshold=freq_diff,
        palindrome_maf_threshold=palindrome_maf,
        verbose=verbose,
    )
    run_check(config)

def main() -> None:
    app()

if __name__ == "__main__":
    main()
```

---

## Testing Requirements

### Unit Test Coverage

| Component | Required Tests |
|-----------|----------------|
| `strand.py` | All 4 strand/allele combinations, palindromic detection, complement function |
| `reference/` | HRC format parsing, 1000G format parsing, population column selection |
| `parsers/` | Valid BIM parsing, malformed lines, missing frequencies |
| `checks/` | Position matching, ID matching, duplicate detection |
| `writers/` | Output file format matches original Perl script |

### Test Patterns

```python
# tests/test_strand.py
import pytest
from imputation_harmonizer.checks.strand import complement, is_palindromic, check_strand_and_alleles
from imputation_harmonizer.models import StrandAction, ExcludeReason

class TestComplement:
    @pytest.mark.parametrize("base,expected", [
        ("A", "T"),
        ("T", "A"),
        ("C", "G"),
        ("G", "C"),
        ("N", "N"),
    ])
    def test_complement_bases(self, base: str, expected: str) -> None:
        assert complement(base) == expected

class TestPalindromic:
    @pytest.mark.parametrize("a1,a2,expected", [
        ("A", "T", True),
        ("T", "A", True),
        ("G", "C", True),
        ("C", "G", True),
        ("A", "G", False),
        ("A", "C", False),
    ])
    def test_is_palindromic(self, a1: str, a2: str, expected: bool) -> None:
        assert is_palindromic(a1, a2) == expected

class TestStrandCheck:
    """Test the 6 possible outcomes from check_strand_and_alleles."""

    def test_case1_strand_ok_refalt_ok(self) -> None:
        """Alleles match exactly - no action needed."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("A", "G"),
            ref_alt_af=0.3,
            bim_af=0.28,
        )
        assert result["exclude"] is False
        assert result["strand_action"] == StrandAction.NONE
        assert result["check_code"] == 1

    def test_case2_strand_ok_refalt_swapped(self) -> None:
        """Alleles swapped - need to force reference allele."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("G", "A"),
            ref_alt_af=0.3,
            bim_af=0.28,
        )
        assert result["exclude"] is False
        assert result["allele_action"] == AlleleAction.FORCE_REF
        assert result["force_ref_allele"] == "A"
        assert result["check_code"] == 2

    def test_case3_strand_flip_refalt_ok(self) -> None:
        """Opposite strand - need to flip."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("T", "C"),
            ref_alt_af=0.3,
            bim_af=0.28,
        )
        assert result["exclude"] is False
        assert result["strand_action"] == StrandAction.FLIP
        assert result["check_code"] == 3

    def test_case5_palindromic_high_maf_excluded(self) -> None:
        """Palindromic SNP with MAF > 0.4 - exclude."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "T"),
            bim_alleles=("A", "T"),
            ref_alt_af=0.45,
            bim_af=0.44,
        )
        assert result["exclude"] is True
        assert result["exclude_reason"] == ExcludeReason.PALINDROMIC_HIGH_MAF
        assert result["check_code"] == 5

    def test_case6_freq_diff_excluded(self) -> None:
        """Large frequency difference - exclude."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("A", "G"),
            ref_alt_af=0.3,
            bim_af=0.05,  # diff > 0.2
        )
        assert result["exclude"] is True
        assert result["exclude_reason"] == ExcludeReason.FREQ_DIFF_TOO_HIGH
        assert result["check_code"] == 6
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
from pathlib import Path
from imputation_harmonizer.main import run_check
from imputation_harmonizer.config import Config

@pytest.fixture
def sample_files(tmp_path: Path) -> dict[str, Path]:
    """Create minimal test files."""
    bim = tmp_path / "test.bim"
    bim.write_text("1\trs123\t0\t10000\tA\tG\n")

    frq = tmp_path / "test.frq"
    frq.write_text("CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n1\trs123\tA\tG\t0.30\t1000\n")

    ref = tmp_path / "ref.tab"
    ref.write_text("#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n1\t10000\trs123\tA\tG\t300\t1000\t0.30\n")

    return {"bim": bim, "frq": frq, "ref": ref, "dir": tmp_path}

def test_full_pipeline_creates_output_files(sample_files: dict[str, Path]) -> None:
    """Test that all expected output files are created."""
    config = Config(
        bim_file=sample_files["bim"],
        frq_file=sample_files["frq"],
        ref_file=sample_files["ref"],
        panel="hrc",
        output_dir=sample_files["dir"],
    )

    run_check(config)

    assert (sample_files["dir"] / "Exclude-test-HRC.txt").exists()
    assert (sample_files["dir"] / "Strand-Flip-test-HRC.txt").exists()
    assert (sample_files["dir"] / "Force-Allele1-test-HRC.txt").exists()
    assert (sample_files["dir"] / "Run-plink.sh").exists()
```

### Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def mini_hrc(fixtures_dir: Path) -> Path:
    """Path to minimal HRC reference file."""
    return fixtures_dir / "mini_hrc.tab"

@pytest.fixture
def mini_1000g(fixtures_dir: Path) -> Path:
    """Path to minimal 1000G reference file."""
    return fixtures_dir / "mini_1000g.legend"
```

---

## Performance Requirements

| Metric | Target | Original Perl |
|--------|--------|---------------|
| **Memory** | <10 GB | ~20 GB |
| **Speed** | 10x faster | Baseline |
| **Reference loading** | <60 seconds | ~120 seconds |

### Performance Patterns

```python
# ✅ GOOD - Stream BIM file, don't load into memory
def parse_bim(filepath: Path) -> Iterator[BimVariant]:
    """Stream variants from BIM file."""
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            yield BimVariant(
                chr=parts[0],
                id=parts[1],
                genetic_dist=float(parts[2]),
                pos=int(parts[3]),
                allele1=parts[4],
                allele2=parts[5],
            )

# ✅ GOOD - Use dict for O(1) lookup
reference: dict[str, ReferenceVariant] = {}
for variant in load_reference(ref_file):
    key = f"{variant.chr}-{variant.pos}"
    reference[key] = variant

# ✅ GOOD - Use slots for memory efficiency
@dataclass(slots=True)
class ReferenceVariant:
    ...

# ❌ BAD - Loading entire file into list
variants = list(parse_bim(filepath))  # Don't do this for large files
```

---

## AI/Claude-Specific Instructions

### Before Writing Code
1. Read `imputation-harmonizer-python-project-plan.md` for full specification
2. Check which module you're implementing against the project structure
3. Review the original Perl script logic for the corresponding section

### Attribution
Mark all substantial AI changes:
```python
# [claude-assisted] Implements strand checking logic from Perl lines 595-723
def check_strand_and_alleles(...):
    ...
```

### Comments for Genomics Logic
Always explain domain-specific assumptions:
```python
# HRC r1 contains only autosomes (chr 1-22), no X/Y/MT
# 1000G includes X chromosome, coded as "23" in some files

# Palindromic SNPs (A/T, G/C) cannot be strand-resolved when MAF > 0.4
# because both strands give the same alleles

# Indels in Illumina format use "-/A" but HRC uses "T/TA"
# These will always fail allele matching - exclude them
```

### Edge Cases to Handle
- Missing frequencies in .frq file (use 0.0 or skip)
- Multiallelic sites in 1000G (marked in TYPE column)
- rsIDs with compound format: `rs123:10177:A:C`
- Position updates creating new duplicates
- Chromosome naming: "1" vs "chr1" vs "01"

---

## What NOT To Do

| Don't | Why |
|-------|-----|
| Use Pydantic BaseModel | Dataclasses are faster for this use case |
| Use async/await | File I/O doesn't benefit; adds complexity |
| Load entire BIM into memory | Files can be 1M+ variants |
| Use FastAPI | This is a CLI tool, not a web service |
| Change output file format | Must match original Perl for validation |
| Skip type hints | Required for all public functions |
| Skip tests | All checking logic needs test coverage |
| Implement multiple modules in one PR | Incremental changes only |

---

## Validation Checklist

Before considering any module complete:

- [ ] `mypy imputation_harmonizer/ --strict` passes
- [ ] `black --check .` passes
- [ ] `ruff check .` passes (or `flake8`)
- [ ] `pytest tests/ -v --cov=imputation_harmonizer` shows >90% coverage
- [ ] Output files match original Perl script format
- [ ] Memory usage tested with large reference panel
- [ ] Progress indicator shown for long operations

---

## Output File Formats

Must match exactly for validation against Perl script:

### Exclude-{stem}-{panel}.txt
```
rs123
rs456
rs789
```

### Strand-Flip-{stem}-{panel}.txt
```
rs111
rs222
```

### Force-Allele1-{stem}-{panel}.txt
```
rs333	A
rs444	G
```

### Position-{stem}-{panel}.txt
```
rs555	12345
rs666	67890
```

### Run-plink.sh
```bash
#!/bin/bash
plink --bfile {stem} --exclude Exclude-{stem}-{panel}.txt --make-bed --out TEMP1
plink --bfile TEMP1 --update-map Chromosome-{stem}-{panel}.txt --update-chr --make-bed --out TEMP2
plink --bfile TEMP2 --update-map Position-{stem}-{panel}.txt --make-bed --out TEMP3
plink --bfile TEMP3 --flip Strand-Flip-{stem}-{panel}.txt --make-bed --out TEMP4
plink --bfile TEMP4 --reference-allele Force-Allele1-{stem}-{panel}.txt --make-bed --out {stem}-updated

for i in {1..22}; do
    plink --bfile {stem}-updated --reference-allele Force-Allele1-{stem}-{panel}.txt --chr $i --make-bed --out {stem}-updated-chr$i
done

rm TEMP*
```

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
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "isort>=5.12.0",
]
fast = [
    "polars>=0.19.0",
]

[project.scripts]
imputation-harmonizer = "imputation_harmonizer.cli:main"

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_ignores = true

[tool.ruff]
line-length = 88
target-version = "py310"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=imputation_harmonizer --cov-report=term-missing"
```

---

## Quick Reference Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all quality checks
black . && isort . && ruff check . && mypy imputation_harmonizer/ && pytest

# Run the tool
imputation-harmonizer -b data.bim -f data.frq -r HRC.r1.sites.tab -p hrc
imputation-harmonizer -b data.bim -f data.frq -r 1000G.legend -p 1000g --pop EUR

# After running, apply corrections
bash Run-plink.sh
```

---

This file must be respected by all AI and human contributors for code quality, scientific integrity, and output compatibility with the original Perl implementation.