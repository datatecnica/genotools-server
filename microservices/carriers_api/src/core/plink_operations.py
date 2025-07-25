import subprocess
from typing import Optional, List
from abc import ABC, abstractmethod


# command interface
class PlinkCommand(ABC):
    @abstractmethod
    def get_command_string(self) -> str:
        """Returns the command string to be executed"""
        pass
    
    def execute(self) -> None:
        """Executes the command"""
        cmd = self.get_command_string()
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            # Handle PLINK2 exit code 13: "Invalid/no variants specified."
            # This typically means no variants remain after --extract
            if e.returncode == 13:
                raise ValueError("No variants found after extraction - this ancestry/chromosome may not contain any target variants")
            else:
                # Re-raise other errors
                raise

# concrete command implementations
class ExtractSnpsCommand(PlinkCommand):
    def __init__(self, pfile: str, snps_file: str, out: str, output_chr: str = None):
        self.pfile = pfile
        self.snps_file = snps_file
        self.out = out
        self.output_chr = output_chr
        
    def get_command_string(self) -> str:
        cmd = f"plink2 --pfile {self.pfile} --extract {self.snps_file}"
        if self.output_chr:
            cmd += f" --output-chr {self.output_chr}"
        cmd += f" --make-pgen psam-cols=-fid --out {self.out}"
        return cmd

class FrequencyCommand(PlinkCommand):
    def __init__(self, pfile: str, out: str):
        self.pfile = pfile
        self.out = out
        
    def get_command_string(self) -> str:
        return f"plink2 --pfile {self.pfile} --freq --out {self.out}"

class SwapAllelesCommand(PlinkCommand):
    def __init__(self, pfile: str, swap_file: str, out: str):
        self.pfile = pfile
        self.swap_file = swap_file
        self.out = out
        
    def get_command_string(self) -> str:
        return f"plink2 --pfile {self.pfile} --a1-allele {self.swap_file} 2 1 --make-pgen --out {self.out}"

class UpdateAllelesCommand(PlinkCommand):
    def __init__(self, pfile: str, update_file: str, out: str):
        self.pfile = pfile
        self.update_file = update_file
        self.out = out
        
    def get_command_string(self) -> str:
        return f"plink2 --pfile {self.pfile} --update-alleles {self.update_file} --make-pgen --out {self.out}"

class ExportCommand(PlinkCommand):
    def __init__(self, pfile: str, out: str, additional_args: Optional[List[str]] = None):
        self.pfile = pfile
        self.out = out
        self.additional_args = additional_args or []
        
    def get_command_string(self) -> str:
        cmd_parts = [
            f"plink2 --pfile {self.pfile}",
            "--export Av",
            "--freq",
            "--missing"
        ]
        cmd_parts.extend(self.additional_args)
        cmd_parts.append(f"--out {self.out}")
        return " ".join(cmd_parts)

class CopyFilesCommand(PlinkCommand):
    def __init__(self, source_prefix: str, target_prefix: str):
        self.source_prefix = source_prefix
        self.target_prefix = target_prefix
        
    def get_command_string(self) -> str:
        return f"cp {self.source_prefix}.pgen {self.target_prefix}.pgen && " \
               f"cp {self.source_prefix}.pvar {self.target_prefix}.pvar && " \
               f"cp {self.source_prefix}.psam {self.target_prefix}.psam"

# class RemoveChrPrefixCommand(PlinkCommand):
#     def __init__(self, pfile: str, out: str):
#         self.pfile = pfile
#         self.out = out
        
#     def get_command_string(self) -> str:
#         return f"plink2 --pfile {self.pfile} --output-chr M --make-pgen --out {self.out}"

# class UpdateChromosomeFormatCommand(PlinkCommand):
#     def __init__(self, pfile: str, out: str, chr_format: str = 'M'):
#         self.pfile = pfile
#         self.out = out
#         self.chr_format = chr_format
        
#     def get_command_string(self) -> str:
#         return f"plink2 --pfile {self.pfile} --output-chr {self.chr_format} --make-pgen --out {self.out}"