from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from src.core.manager import CarrierAnalysisManager
from src.core.recruitment_manager import create_recruitment_analyzer
from src.core.security import get_api_key

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class CarrierRequest(BaseModel):
    geno_path: str  # Path to PLINK2 files prefix (without .pgen/.pvar/.psam extension)
    snplist_path: str  # Path to SNP list file
    out_prefix: str  # Output file prefix (directory will be created as needed)
    release_version: str = "10"  # Default release version
    # labels: Optional[List[str]] = ['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN', 'MDE', 'SAS']

class RecruitmentAnalysisRequest(BaseModel):
    release: str  = "10"  # GP2 release version (e.g., "10", "9")
    # Optional direct input path overrides (generic naming for any carrier source)
    key_path: Optional[str] = None
    dict_path: Optional[str] = None
    ext_clin_path: Optional[str] = None
    carriers_var_info: Optional[str] = None
    carriers_int: Optional[str] = None
    carriers_string: Optional[str] = None
    # Optional output override
    output_dir: Optional[str] = None
    # Dry run flag
    dry_run: bool = False  # Check paths without running analysis
class ImputedCarrierRequest(BaseModel):
    ancestry: str  # Ancestry label (e.g., 'AAC', 'AFR')
    imputed_dir: str  # Base directory for imputed genotypes
    snplist_path: str  # Path to SNP list file
    out_path: str  # Full output path prefix for the generated files
    release_version: str = "10"  # Default release version

@app.post("/process_carriers")
async def process_carriers(
    request: CarrierRequest,
    # api_key: str = Depends(get_api_key)
):
    """
    Process carrier information from genotype files.
    Handles both single files (NBA/WGS) and chromosome-split files (imputed) automatically.
    Returns paths to the generated files.
    """
    try:
        parent_dir = os.path.dirname(request.out_prefix)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        manager = CarrierAnalysisManager()
        
        results = manager.extract_carriers(
            geno_path=request.geno_path,
            snplist_path=request.snplist_path,
            out_path=request.out_prefix
        )

        return {
            "status": "success",
            "outputs": results
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {str(e)}\n\nTraceback: {error_trace}"
        )

@app.post("/recruitment_analysis")
async def run_recruitment_analysis(
    request: RecruitmentAnalysisRequest,
    # api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Run precision medicine recruitment analysis for genetic carriers.
    
    Analyzes clinical trial recruitment potential by evaluating genetic carriers
    and generating recruitment statistics by ancestry and study cohort.
    
    Returns paths to generated analysis files and summary statistics.
    """
    try:
        # Expand output override if provided
        output_dir = os.path.expanduser(request.output_dir) if request.output_dir else None

        # Create analyzer using config defaults (mnt_path derived internally)
        analyzer = create_recruitment_analyzer(
            release=request.release,
            output_dir=output_dir
        )
        
        # Check if required data paths exist
        clinical_repo = analyzer.clinical_repo
        carrier_repo = analyzer.carrier_repo
        
        # Apply direct-path overrides (generic carriers_* naming)
        if request.key_path:
            clinical_repo.paths['key_path'] = os.path.expanduser(request.key_path)
        if request.dict_path:
            clinical_repo.paths['dict_path'] = os.path.expanduser(request.dict_path)
        if request.ext_clin_path:
            clinical_repo.paths['ext_clin_path'] = os.path.expanduser(request.ext_clin_path)
        if request.carriers_var_info:
            carrier_repo.paths['wgs_var_info'] = os.path.expanduser(request.carriers_var_info)
        if request.carriers_int:
            carrier_repo.paths['wgs_int'] = os.path.expanduser(request.carriers_int)
        if request.carriers_string:
            carrier_repo.paths['wgs_string'] = os.path.expanduser(request.carriers_string)

        paths_to_check = [
            ('Master Key', clinical_repo.paths['key_path']),
            ('Data Dictionary', clinical_repo.paths['dict_path']),
            ('Extended Clinical', clinical_repo.paths['ext_clin_path']),
            ('WGS Variant Info', carrier_repo.paths['wgs_var_info']),
            ('WGS Carriers Int', carrier_repo.paths['wgs_int']),
            ('WGS Carriers String', carrier_repo.paths['wgs_string'])
        ]
        
        missing_paths = []
        for name, path in paths_to_check:
            expanded_path = os.path.expanduser(path)
            if not os.path.exists(expanded_path):
                missing_paths.append(f"{name}: {expanded_path}")
        
        if missing_paths:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required data files: {missing_paths}"
            )
        
        # If dry run, just return path validation results
        if request.dry_run:
            return {
                "status": "dry_run_success",
                "message": "All required data paths found",
                "config": {
                    "release": request.release,
                    "mnt_path": analyzer.config.mnt_path,
                    "output_dir": analyzer.config.output_dir
                },
                "validated_paths": [name for name, _ in paths_to_check]
            }
        
        # Run full analysis
        results = analyzer.run_full_analysis()
        
        # Extract key statistics for response
        summary_stats = {}
        for locus in analyzer.carrier_data.keys():
            summary_stats[locus] = {
                "total_carriers": len(analyzer.carrier_data[locus]),
                "recruitment_analyzed": f"{locus}_recruitment" in results
            }
        
        # Count exported files
        exported_files_count = len(results.get('exported_files', {}))
        
        return {
            "status": "success",
            "message": "Recruitment analysis completed successfully",
            "config": {
                "release": request.release,
                "mnt_path": analyzer.config.mnt_path,
                "output_dir": analyzer.config.output_dir
            },
            "summary": {
                "loci_analyzed": list(analyzer.carrier_data.keys()),
                "total_loci": len(analyzer.carrier_data),
                "carrier_statistics": summary_stats,
                "exported_files_count": exported_files_count
            },
            "output_directory": analyzer.config.output_dir,
            "exported_files": results.get('exported_files', {})
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
@app.post("/process_imputed_carriers")
async def process_imputed_carriers(
    request: ImputedCarrierRequest,
    # api_key: str = Depends(get_api_key)
):
    """
    Process imputed carrier information from chromosome-split genotype files.
    This endpoint is kept for backwards compatibility but uses the same unified processing.
    Returns paths to the generated files.
    """
    try:
        parent_dir = os.path.dirname(request.out_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        manager = CarrierAnalysisManager()
        
        results = manager.extract_carriers(
            geno_path=request.imputed_dir,
            snplist_path=request.snplist_path,
            out_path=request.out_path,
            ancestry=request.ancestry,
            release=request.release_version
        )

        return {
            "status": "success",
            "ancestry": request.ancestry,
            "outputs": results
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {str(e)}\n\nTraceback: {error_trace}"
        )
