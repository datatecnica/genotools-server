from fastapi import APIRouter, HTTPException, Depends, status, Response
from services.data_service import load_cohort_data_from_csv,load_cohort_data_from_df
from models.data_models import CohortDataSchema
from fastapi.security.api_key import APIKeyHeader
from typing import List
from functools import lru_cache
from utils import init_db
import os
import json

import logging
# from google.cloud import secretmanager #, storage

logger = logging.getLogger(__name__)

router = APIRouter()

# with open('/app/secretname.txt', 'r') as f:
#     secret_value = f.read().strip()

# secret_data = json.loads('/var/secrets/secretname.txt')
# secret_data = '/var/secrets/secretname.txt'
# print(f"secret_data: {secret_data}")
# print(f"GENOTRACKER_API_KEY: {secret_data['GENOTRACKER_API_KEY']}")
# print(f"GENOTOOLS_API_KEY: {secret_data['GENOTOOLS_API_KEY']}")
# print(f"EMAIL_PAT: {secret_data['EMAIL_PAT']}")

# Retrieve the API key from Secret Manager
API_KEY = os.environ["GTRACKER_API"] 

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid API Key",
        )
    return api_key

# @st.cache_resource(show_spinner=False)
@lru_cache()
def init_db_connection():
    return init_db()
# Initialize the database connection and read data
df = init_db_connection()



@router.get("/")
async def root():
    return f"Welcome to GenoTracker"

@router.get("/data", response_model=List[CohortDataSchema])
def get_cohort_data(
    response: Response,
    from_gcs: bool = False, 
    api_key: str = Depends(get_api_key)
    ):
    try:
        return load_cohort_data_from_df(df)
    except Exception as e:
        logger.error(f"Error loading cohort data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
