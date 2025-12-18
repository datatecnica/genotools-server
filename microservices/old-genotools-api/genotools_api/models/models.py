from pydantic import BaseModel, field_validator
from typing import Optional, Union

class GenoToolsParams(BaseModel):
    email: Optional[str] = None
    bfile: Optional[str] = None
    pfile: Optional[str] = None
    vcf: Optional[str] = None
    out: Optional[str] = None
    full_output: Optional[bool] = None
    skip_fails: Optional[bool] = None
    warn: Optional[bool] = None
    # can use the following for others that take a value but also have a default in the package
    callrate: Optional[Union[bool, float]] = None
    sex: Optional[bool] = None
    related: Optional[bool] = None
    related_cutoff: Optional[float] = None
    duplicated_cutoff: Optional[float] = None
    prune_related: Optional[bool] = None
    prune_duplicated: Optional[bool] = None
    het: Optional[bool] = None
    all_sample: Optional[bool] = None
    all_variant: Optional[bool] = None
    maf: Optional[float] = None
    ancestry: Optional[bool] = None
    ref_panel: Optional[str] = None
    ref_labels: Optional[str] = None
    model: Optional[str] = None
    storage_type: str = 'local'
    
    @field_validator('model', 'ref_panel', 'ref_labels')
    def none_string_to_none(cls, v):
        if v == 'None':
            return None
        return v