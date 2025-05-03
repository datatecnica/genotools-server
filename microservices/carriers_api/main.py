from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import os
from typing import List, Optional
from src.core.manager import CarrierAnalysisManager
from src.core.security import get_api_key

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class CarrierRequest(BaseModel):
    geno_path: str  # Path to PLINK2 files prefix (without .pgen/.pvar/.psam extension)
    key_file_path: str  # Path to key file
    snplist_path: str  # Path to SNP list file
    out_path: str  # Full output path prefix for the generated files
    release_version: str = "9"  # Default release version
    # labels: Optional[List[str]] = ['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN', 'MDE', 'SAS']

@app.post("/process_carriers")
async def process_carriers(
    request: CarrierRequest,
    # api_key: str = Depends(get_api_key)
):
    """
    Process carrier information from a single genotype file stored locally.
    Returns paths to the generated file.
    """
    try:
        parent_dir = os.path.dirname(request.out_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        manager = CarrierAnalysisManager()
        
        results = manager.extract_carriers(
            geno_path=request.geno_path,
            snplist_path=request.snplist_path,
            out_path=request.out_path
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

# if __name__ == "__main__":