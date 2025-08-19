from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .variant import VariantList
from .carrier import Carrier, CarrierReport


class DataType(str, Enum):
    NBA = "NBA"
    WGS = "WGS"
    IMPUTED = "IMPUTED"
    
    @property
    def description(self) -> str:
        descriptions = {
            "NBA": "NeuroBooster Array - Split by ancestry",
            "WGS": "Whole Genome Sequencing - Single consolidated file",
            "IMPUTED": "Imputed genotypes - Split by ancestry and chromosome"
        }
        return descriptions.get(self.value, "Unknown data type")


class AnalysisStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    
    @property
    def is_terminal(self) -> bool:
        return self in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED)
    
    @property
    def is_active(self) -> bool:
        return self in (AnalysisStatus.PENDING, AnalysisStatus.PROCESSING)


class AnalysisRequest(BaseModel):
    variant_list: VariantList = Field(..., description="List of variants to analyze")
    data_type: DataType = Field(..., description="Type of genomic data to analyze")
    release: str = Field(..., description="GP2 release version (e.g., '10')")
    ancestries: Optional[List[str]] = Field(
        None,
        description="List of ancestries to analyze. If None, all available ancestries"
    )
    chromosomes: Optional[List[str]] = Field(
        None,
        description="List of chromosomes to analyze (for IMPUTED data). If None, all chromosomes"
    )
    include_clinical_data: bool = Field(
        True,
        description="Whether to include clinical data in carrier reports"
    )
    output_format: str = Field(
        "parquet",
        description="Output format for results (parquet, json, csv)"
    )
    
    @field_validator('release')
    @classmethod
    def validate_release(cls, v: str) -> str:
        try:
            release_num = int(v)
            if release_num < 1 or release_num > 100:
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid release: {v}. Must be a valid release number")
        return v
    
    @field_validator('ancestries')
    @classmethod
    def validate_ancestries(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        valid_ancestries = ["AAC", "AJ", "CAH", "CAS", "EAS", "EUR", "FIN", "LAS", "MDE", "SAS", "SSA"]
        if v:
            invalid = [a for a in v if a not in valid_ancestries]
            if invalid:
                raise ValueError(f"Invalid ancestries: {invalid}. Must be one of {valid_ancestries}")
        return v
    
    @field_validator('chromosomes')
    @classmethod
    def validate_chromosomes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        valid_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
        if v:
            invalid = [c for c in v if c not in valid_chroms]
            if invalid:
                raise ValueError(f"Invalid chromosomes: {invalid}. Must be one of {valid_chroms}")
        return v
    
    @field_validator('output_format')
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        valid_formats = ['parquet', 'json', 'csv']
        if v not in valid_formats:
            raise ValueError(f"Invalid output format: {v}. Must be one of {valid_formats}")
        return v
    
    @property
    def requires_chromosome_split(self) -> bool:
        return self.data_type == DataType.IMPUTED
    
    @property
    def requires_ancestry_split(self) -> bool:
        return self.data_type in (DataType.NBA, DataType.IMPUTED)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "variant_list": {
                    "name": "Pathogenic Panel",
                    "variants": [
                        {
                            "chromosome": "1",
                            "position": 762273,
                            "ref": "G",
                            "alt": "A",
                            "gene": "BRCA2",
                            "rsid": "rs121913023",
                            "inheritance_pattern": "AD"
                        }
                    ]
                },
                "data_type": "NBA",
                "release": "10",
                "ancestries": ["EUR", "EAS"],
                "include_clinical_data": True,
                "output_format": "parquet"
            }
        }
    )


class AnalysisMetadata(BaseModel):
    start_time: datetime = Field(..., description="Analysis start time")
    end_time: Optional[datetime] = Field(None, description="Analysis end time")
    duration_seconds: Optional[float] = Field(None, description="Analysis duration in seconds")
    files_processed: int = Field(0, description="Number of files processed")
    samples_analyzed: int = Field(0, description="Total samples analyzed")
    variants_analyzed: int = Field(0, description="Total variants analyzed")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    processing_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing statistics"
    )
    
    def set_end_time(self):
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_time": "2024-01-01T10:00:00",
                "end_time": "2024-01-01T10:30:00",
                "duration_seconds": 1800,
                "files_processed": 11,
                "samples_analyzed": 50000,
                "variants_analyzed": 400,
                "errors": [],
                "warnings": ["Missing clinical data for 10 samples"],
                "processing_stats": {
                    "cache_hits": 350,
                    "cache_misses": 50,
                    "memory_peak_gb": 8.5
                }
            }
        }
    )


class AnalysisResult(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: AnalysisStatus = Field(..., description="Current analysis status")
    request: AnalysisRequest = Field(..., description="Original analysis request")
    carrier_reports: List[CarrierReport] = Field(
        default_factory=list,
        description="Carrier reports for each variant"
    )
    summary_statistics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics across all variants"
    )
    metadata: AnalysisMetadata = Field(..., description="Analysis metadata")
    output_files: List[str] = Field(
        default_factory=list,
        description="List of generated output files"
    )
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")
    
    @property
    def is_complete(self) -> bool:
        return self.status == AnalysisStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        return self.status == AnalysisStatus.FAILED
    
    @property
    def total_carriers_found(self) -> int:
        return sum(report.total_carrier_count for report in self.carrier_reports)
    
    @property
    def variants_with_carriers(self) -> int:
        return sum(1 for report in self.carrier_reports if report.total_carrier_count > 0)
    
    def get_report_for_variant(self, variant_id: str) -> Optional[CarrierReport]:
        for report in self.carrier_reports:
            if report.variant_id == variant_id:
                return report
        return None
    
    def add_error(self, error: str):
        self.metadata.errors.append(error)
        self.status = AnalysisStatus.FAILED
        self.error_message = error
    
    def add_warning(self, warning: str):
        self.metadata.warnings.append(warning)
    
    def complete_analysis(self):
        self.status = AnalysisStatus.COMPLETED
        self.metadata.set_end_time()
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "analysis_20240101_100000_abc123",
                "status": "COMPLETED",
                "request": {
                    "data_type": "NBA",
                    "release": "10",
                    "ancestries": ["EUR"],
                    "output_format": "parquet"
                },
                "carrier_reports": [],
                "summary_statistics": {
                    "total_variants": 400,
                    "variants_with_carriers": 350,
                    "total_carriers": 5000,
                    "average_carrier_frequency": 0.012
                },
                "metadata": {
                    "start_time": "2024-01-01T10:00:00",
                    "end_time": "2024-01-01T10:30:00",
                    "duration_seconds": 1800,
                    "files_processed": 11,
                    "samples_analyzed": 50000,
                    "variants_analyzed": 400
                },
                "output_files": [
                    "~/gcs_mounts/genotools_server/carriers/results/analysis_20240101_100000_abc123/carrier_report.parquet",
                    "~/gcs_mounts/genotools_server/carriers/results/analysis_20240101_100000_abc123/summary_stats.parquet"
                ]
            }
        }
    )