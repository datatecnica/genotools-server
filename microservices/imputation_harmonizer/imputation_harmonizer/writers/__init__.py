"""Output file writers for PLINK update files and shell scripts."""

from imputation_harmonizer.writers.plink_files import PlinkFileWriter
from imputation_harmonizer.writers.shell_script import write_shell_script

__all__ = ["PlinkFileWriter", "write_shell_script"]
