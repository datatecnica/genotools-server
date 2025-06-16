#!/bin/bash
export ZONE=europe-west4-a
export GCP_PROJECT=gp2-release-terra
# export GCP_PROJECT=gp2-code-test-env 
export CLUSTER_NAME=gtserver-eu-west4
export GENOTOOLS_API_NODE_POOL=genotools-api-ancesstory-pool
export VM=c4-highmem-32

echo setting gcp project: $GCP_PROJECT

# gcloud config set project $GCP_PROJECT

echo Adding node-pool: ${GENOTOOLS_API_NODE_POOL} with VM: $VM to use with the genotools-api deployment on $CLUSTER_NAME, k8s cluster in zone: $ZONE -  Please change machine type if needed

gcloud container node-pools create ${GENOTOOLS_API_NODE_POOL} \
  --cluster=$CLUSTER_NAME \
  --machine-type=$VM \
  --zone=$ZONE \
  --num-nodes=1
