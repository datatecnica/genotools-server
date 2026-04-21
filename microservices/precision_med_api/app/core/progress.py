"""
Enhanced progress tracking for precision medicine pipeline.

Provides clean, informative progress output using tqdm with:
- Pipeline step tracking (Step 1/4, Step 2/4, etc.)
- Per-data-type breakdown during extraction
- Dynamic ETA based on actual processing speed
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tqdm import tqdm


@dataclass
class DataTypeProgress:
    """Track progress for a single data type."""
    total: int = 0
    completed: int = 0
    empty: int = 0
    failed: int = 0


@dataclass
class PipelineProgress:
    """
    Track overall pipeline progress with enhanced status display.

    Usage:
        progress = PipelineProgress(job_name="release11", total_steps=4)
        progress.start_step(1, "Loading SNP list")
        # ... do work ...
        progress.complete_step("Loaded 400 variants")

        progress.start_step(2, "Extracting variants")
        with progress.extraction_bar(total_files=150, data_types=["NBA", "WGS"]) as bar:
            for file in files:
                # process file
                bar.update("NBA")  # or bar.update("WGS", empty=True)
    """

    job_name: str
    total_steps: int = 4
    current_step: int = 0
    start_time: float = field(default_factory=time.time)
    step_start_time: float = field(default_factory=time.time)

    def __post_init__(self):
        """Print pipeline header."""
        print(f"\nPrecision Medicine Pipeline - {self.job_name}")
        print("=" * 50)
        print()

    def start_step(self, step_num: int, description: str):
        """Announce the start of a pipeline step."""
        self.current_step = step_num
        self.step_start_time = time.time()
        print(f"Step {step_num}/{self.total_steps}: {description}")

    def complete_step(self, summary: Optional[str] = None):
        """Mark current step as complete with optional summary."""
        elapsed = time.time() - self.step_start_time
        if summary:
            print(f"  {summary} ({elapsed:.1f}s)")
        print()

    def log_substep(self, message: str):
        """Log a substep message (indented)."""
        print(f"  {message}")

    def extraction_bar(
        self,
        total_files: int,
        data_types: List[str],
        planned_by_type: Optional[Dict[str, int]] = None
    ) -> 'ExtractionProgressBar':
        """
        Create an extraction progress bar with per-data-type tracking.

        Args:
            total_files: Total number of files to process
            data_types: List of data type names (e.g., ["NBA", "WGS", "IMPUTED"])
            planned_by_type: Optional dict of planned files per type

        Returns:
            ExtractionProgressBar context manager
        """
        return ExtractionProgressBar(
            total_files=total_files,
            data_types=data_types,
            planned_by_type=planned_by_type
        )

    def finish(self):
        """Print final summary."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        if minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        print(f"Pipeline completed in {time_str}")


class ExtractionProgressBar:
    """
    Progress bar for variant extraction with per-data-type breakdown.

    Shows:
    - Overall progress bar with ETA
    - Per-data-type counts (NBA: 45/50, WGS: 30/40, etc.)
    """

    def __init__(
        self,
        total_files: int,
        data_types: List[str],
        planned_by_type: Optional[Dict[str, int]] = None
    ):
        self.total_files = total_files
        self.data_types = data_types
        self.planned_by_type = planned_by_type or {}

        # Initialize per-type tracking
        self.progress: Dict[str, DataTypeProgress] = {}
        for dt in data_types:
            self.progress[dt] = DataTypeProgress(
                total=self.planned_by_type.get(dt, 0)
            )

        self.pbar: Optional[tqdm] = None

    def __enter__(self) -> 'ExtractionProgressBar':
        """Start the progress bar."""
        self.pbar = tqdm(
            total=self.total_files,
            desc="  Extracting",
            unit="file",
            file=sys.stdout,
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the progress bar and print summary."""
        if self.pbar:
            self.pbar.close()

        # Print per-data-type summary
        summary_parts = []
        for dt in self.data_types:
            p = self.progress[dt]
            if p.total > 0:
                summary_parts.append(f"{dt}: {p.completed}/{p.total}")

        if summary_parts:
            print(f"  {' | '.join(summary_parts)}")

        return False

    def update(
        self,
        data_type: str,
        success: bool = True,
        empty: bool = False
    ):
        """
        Update progress for a completed file.

        Args:
            data_type: The data type of the completed file
            success: Whether extraction succeeded (False = failed)
            empty: Whether result was empty (still counts as processed)
        """
        if data_type in self.progress:
            p = self.progress[data_type]
            if success:
                if empty:
                    p.empty += 1
                else:
                    p.completed += 1
            else:
                p.failed += 1

        if self.pbar:
            self.pbar.update(1)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
