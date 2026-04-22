#!/bin/bash
export ZONE=europe-west4-a
export GCP_PROJECT=gp2-release-terra
# export GCP_PROJECT=gp2-code-test-env 
export CLUSTER_NAME=gtserver-eu-west4
export GENOTOOLS_API_NODE_POOL=genotools-api-ancesstory-pool
export VM=c4-highmem-32

echo setting gcp project: $GCP_PROJECT

# gcloud config set project $GCP_PROJECT

echo getting credentials for the cluster: $CLUSTER_NAME in zone: $ZONE

# gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE

echo Now Deleting node-pool: $GENOTOOLS_API_NODE_POOL with VM: $VM for genotools-api from $CLUSTER_NAME, k8s cluster in zone: $ZONE

gcloud container node-pools delete $GENOTOOLS_API_NODE_POOL \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --quiet