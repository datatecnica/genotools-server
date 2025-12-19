"""I/O utilities for transparent gzip handling.

[claude-assisted] Provides smart file opening that auto-detects gzip compression
by checking magic bytes, enabling transparent reading of .gz and uncompressed files.

Example:
    with smart_open(Path("reference.tab.gz")) as f:
        for line in f:
            process(line)
"""

import gzip
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, Literal

# Gzip magic bytes (first two bytes of gzip file)
GZIP_MAGIC = b"\x1f\x8b"


def is_gzipped(filepath: Path) -> bool:
    """Detect if a file is gzip-compressed.

    Checks magic bytes first (reliable), falls back to extension if file
    is too small or unreadable.

    Args:
        filepath: Path to file to check

    Returns:
        True if file is gzip-compressed

    Example:
        >>> is_gzipped(Path("data.tab.gz"))
        True
        >>> is_gzipped(Path("data.tab"))
        False
    """
    try:
        with open(filepath, "rb") as f:
            magic = f.read(2)
            if len(magic) >= 2:
                return magic == GZIP_MAGIC
    except OSError:
        pass

    # Fall back to extension check
    return filepath.suffix == ".gz" or str(filepath).endswith(".gz")


@contextmanager
def smart_open(
    filepath: Path,
    mode: Literal["r", "rt", "rb"] = "rt",
) -> Iterator[IO[str] | IO[bytes]]:
    """Open a file with automatic gzip detection.

    Transparently handles both gzipped and uncompressed files,
    detecting compression by checking magic bytes.

    Args:
        filepath: Path to file (may be .gz or uncompressed)
        mode: File mode ('r' or 'rt' for text, 'rb' for binary)

    Yields:
        File handle (text or binary based on mode)

    Example:
        # Works for both compressed and uncompressed files
        with smart_open(Path("ref.tab.gz")) as f:
            for line in f:
                process(line)

        with smart_open(Path("ref.tab")) as f:
            for line in f:
                process(line)
    """
    # Normalize mode
    if mode == "r":
        mode = "rt"

    gzipped = is_gzipped(filepath)

    if gzipped:
        # Use gzip.open for compressed files
        # For text mode, specify encoding
        if mode == "rt":
            f = gzip.open(filepath, mode, encoding="utf-8")
        else:
            f = gzip.open(filepath, mode)
    else:
        # Use regular open for uncompressed files
        if mode == "rt":
            f = open(filepath, mode, encoding="utf-8")
        else:
            f = open(filepath, mode)

    try:
        yield f
    finally:
        f.close()


def iter_lines(filepath: Path) -> Iterator[str]:
    """Iterate over lines in a file with gzip auto-detection.

    Memory-efficient line iteration for large files. Lines are
    stripped of trailing newlines.

    Args:
        filepath: Path to file (may be gzipped)

    Yields:
        Lines from file (stripped of trailing newlines)

    Example:
        for line in iter_lines(Path("large_file.tab.gz")):
            parts = line.split("\t")
    """
    with smart_open(filepath, "rt") as f:
        for line in f:
            yield line.rstrip("\n")


def count_lines(filepath: Path) -> int:
    """Count lines in a file with gzip auto-detection.

    Efficient line counting without loading entire file into memory.

    Args:
        filepath: Path to file (may be gzipped)

    Returns:
        Number of lines in file

    Example:
        >>> count_lines(Path("variants.bim"))
        1000000
    """
    count = 0
    with smart_open(filepath, "rt") as f:
        for _ in f:
            count += 1
    return count
