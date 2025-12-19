"""Tests for io_utils module (gzip support)."""

import gzip
from pathlib import Path

import pytest

from imputation_harmonizer.io_utils import (
    count_lines,
    is_gzipped,
    iter_lines,
    smart_open,
)


class TestIsGzipped:
    """Test gzip file detection."""

    def test_gzipped_file(self, tmp_path: Path) -> None:
        """Detect gzipped file by magic bytes."""
        gz_file = tmp_path / "test.gz"
        with gzip.open(gz_file, "wt") as f:
            f.write("test content\n")

        assert is_gzipped(gz_file) is True

    def test_plain_file(self, tmp_path: Path) -> None:
        """Detect plain text file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test content\n")

        assert is_gzipped(txt_file) is False

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file is not gzipped."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        assert is_gzipped(empty_file) is False

    def test_gz_extension_but_not_gzipped(self, tmp_path: Path) -> None:
        """File with .gz extension but not actually gzipped."""
        fake_gz = tmp_path / "fake.gz"
        fake_gz.write_text("not gzipped content")

        # Should detect by magic bytes, not extension
        assert is_gzipped(fake_gz) is False


class TestSmartOpen:
    """Test smart_open for automatic gzip handling."""

    def test_open_plain_text(self, tmp_path: Path) -> None:
        """Open plain text file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("line1\nline2\nline3\n")

        with smart_open(txt_file) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert lines[0].strip() == "line1"

    def test_open_gzipped(self, tmp_path: Path) -> None:
        """Open gzipped file transparently."""
        gz_file = tmp_path / "test.gz"
        with gzip.open(gz_file, "wt") as f:
            f.write("line1\nline2\nline3\n")

        with smart_open(gz_file) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert lines[0].strip() == "line1"

    def test_open_binary_mode(self, tmp_path: Path) -> None:
        """Open in binary mode."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_bytes(b"binary content")

        with smart_open(txt_file, mode="rb") as f:
            content = f.read()

        assert content == b"binary content"


class TestIterLines:
    """Test line iterator."""

    def test_iter_all_lines(self, tmp_path: Path) -> None:
        """Iterate over all lines."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("header\nline1\nline2\n")

        lines = list(iter_lines(txt_file))

        assert len(lines) == 3
        # Lines are stripped of trailing newlines
        assert lines[0] == "header"

    def test_iter_strips_newlines(self, tmp_path: Path) -> None:
        """Lines are stripped of trailing newlines."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("line1\nline2\n")

        lines = list(iter_lines(txt_file))

        assert len(lines) == 2
        assert lines[0] == "line1"
        assert lines[1] == "line2"

    def test_iter_gzipped(self, tmp_path: Path) -> None:
        """Iterate lines in gzipped file."""
        gz_file = tmp_path / "test.gz"
        with gzip.open(gz_file, "wt") as f:
            f.write("header\nline1\nline2\n")

        lines = list(iter_lines(gz_file))

        assert len(lines) == 3
        assert lines[0] == "header"


class TestCountLines:
    """Test line counting."""

    def test_count_plain_text(self, tmp_path: Path) -> None:
        """Count lines in plain text file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("line1\nline2\nline3\n")

        assert count_lines(txt_file) == 3

    def test_count_gzipped(self, tmp_path: Path) -> None:
        """Count lines in gzipped file."""
        gz_file = tmp_path / "test.gz"
        with gzip.open(gz_file, "wt") as f:
            f.write("line1\nline2\nline3\n")

        assert count_lines(gz_file) == 3

    def test_count_empty_file(self, tmp_path: Path) -> None:
        """Count lines in empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        assert count_lines(empty_file) == 0

    def test_count_single_line_no_newline(self, tmp_path: Path) -> None:
        """Count single line without trailing newline."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("single line")

        assert count_lines(txt_file) == 1
