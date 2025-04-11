#!/bin/bash
export ZONE=europe-west4-a
export GCP_PROJECT=gp2-code-test-env
export CLUSTER_NAME=gtserver-eu-west4
export GP2_BROWSER_APP_NODE_POOL=gp2-browser-app-node-pool
export VM=e2-standard-4

echo setting gcp project: $GCP_PROJECT

# gcloud config set project $GCP_PROJECT

echo getting credentials for the cluster: $CLUSTER_NAME in zone: $ZONE

# gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE

echo Now Deleting node-pool: $GP2_BROWSER_APP_NODE_POOL with VM: $VM for genotools-api from $CLUSTER_NAME, k8s cluster in zone: $ZONE

gcloud container node-pools delete $GP2_BROWSER_APP_NODE_POOL \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --quiet