from genotools.utils import shell_do
# from google.cloud import storage
import os
from genotools_api.models.models import GenoToolsParams


def download_from_gcs(gcs_path, local_path):
    storage_client = storage.Client()
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path


def upload_to_gcs(local_path, gcs_path):
    storage_client = storage.Client()
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def execute_genotools(command: str, run_locally: bool = True):

    if run_locally:
        return shell_do(command, log=True, return_log=True)
    else:
        return {"message": "GKE execution method to be implemented"}


def expand_path(path: str) -> str:
    """Expand the user path."""
    return os.path.expanduser(path)


def construct_command(params: GenoToolsParams) -> str:
    command = "genotools"
    
    # Mapping options to their respective values in params
    options_with_values = {
        "--callrate": params.callrate,
        "--related_cutoff": params.related_cutoff,
        "--duplicated_cutoff": params.duplicated_cutoff,
        "--maf": params.maf,
        "--ref_panel": params.ref_panel,
        "--ref_labels": params.ref_labels,
        "--pca": params.pca,
        "--geno": params.geno,
        "--case_control": params.case_control,
        "--haplotype": params.haplotype,
        "--hwe": params.hwe,
        "--build": params.build,
        "--covars": params.covars,
        "--covar_names": params.covar_names,
        "--min_samples": params.min_samples,
        "--model": params.model
    }


    flags = [
        ("--full_output", params.full_output),
        ("--skip_fails", params.skip_fails),
        ("--warn", params.warn),
        ("--sex", params.sex),
        ("--related", params.related),
        ("--prune_related", params.prune_related),
        ("--prune_duplicated", params.prune_duplicated),
        ("--het", params.het),
        ("--all_sample", params.all_sample),
        ("--all_variant", params.all_variant),
        ("--gwas", params.gwas),
        ("--amr_het", params.amr_het),
        ("--kinship_check", params.kinship_check),
        ("--filter_controls", params.filter_controls),
        ("--maf_lambdas", params.maf_lambdas),
        ("--subset_ancestry", params.subset_ancestry),
        ("--ancestry", params.ancestry)
    ]

    for option, value in options_with_values.items():
        if value is not None:
            command += f" {option} {value}"
    
    for flag, value in flags:
        if value is not None:
            command += f" {flag} {value}"

    if params.pfile:
        command += f" --pfile {expand_path(params.pfile)}"
    elif params.bfile:
        command += f" --bfile {expand_path(params.bfile)}"
    elif params.vcf:
        command += f" --vcf {expand_path(params.vcf)}"
    else:
        raise ValueError("No geno file provided")

    if params.out:
        command += f" --out {expand_path(params.out)}"
    else:
        raise ValueError("No output file provided")

    return command