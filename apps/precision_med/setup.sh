#!/bin/bash

# Mount the necessary Google Cloud Storage buckets
# These buckets are used for storing and accessing data for the project
# The --implicit-dirs flag allows for the creation of directories within the mounted bucket
gcsfuse --implicit-dirs gp2tier2_vwb ~/gcs_mounts/gp2tier2_vwb
gcsfuse --implicit-dirs genotools-server ~/gcs_mounts/genotools_server
gcsfuse --implicit-dirs gp2_release10_staging ~/gcs_mounts/gp2_release10_staging
