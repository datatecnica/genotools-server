from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import os
from src.core.browser_prep import prep_browser_files
from src.core.security import get_api_key
from src.core.browser_prep import BrowserPrep

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class BrowserRequest(BaseModel):
    release_num: int = 10 # Default release number
    master_path: str # Path to Master Key
    gt_path: str # GCS path to GenoTools outputs (JSON)
    out_dir: str # Output directory for final outputs in genotools-server bucket

@app.post("/prep_browser")
async def prep_browser(
    request: BrowserRequest,
    # api_key: str = Depends(get_api_key)
):
    """
    Process Master Key and GenoTools outputs stored in GCS.
    Returns paths to the generated files in GCS.
    """
    try:
        final_files = prep_browser_files(
            rel=request.release_num,
            master_key=request.master_path,
            gt_output=request.gt_path,
            out_dir=request.out_dir,
        )

        final_gcs_paths = []
        for filename in final_files:
            folders = request.out_dir.strip(os.sep).split(os.sep)
            mount_folder = os.sep.join(folders[-3:])
            gcs_path = f"{mount_folder}/{filename}"
            final_gcs_paths.append(f"gs://{gcs_path}")

        return {
            "status": "success",
            "outputs": final_gcs_paths
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {str(e)}\n\nTraceback: {error_trace}"
        )