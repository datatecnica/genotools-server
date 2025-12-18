#!/bin/bash
# ZONE=europe-west4-a
# REGION=europe-west4
# CLUSTER_NAME=gke-test-cluster
# #PROJECT_NUMBER=719002041197
# PROJECT_ID=gp2-testing-475115
# VM=e2-standard-8
# IDAT_PED_BED_MERGE_NODE_POOL=workflow-idat-ped-bed-nodepool

export ZONE=europe-west4-a
export PROJECT_ID=gp2-release-terra
export CLUSTER_NAME=gke-prod-cluster
export IDAT_PED_BED_MERGE_NODE_POOL=workflow-idat-ped-bed-nodepool
export VM=e2-standard-8


echo setting gcp project: $PROJECT_ID

gcloud config set project $PROJECT_ID

echo retrieving Auth Credentials for the created Cluster: $CLUSTER_NAME, in zone: $ZONE

gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --dns-endpoint


echo Adding node-pool: $IDAT_PED_BED_MERGE_NODE_POOL to use with the gp2_browser app deployment on $CLUSTER_NAME k8s cluster in zone: $ZONE -  Please change machine type if needed


gcloud container node-pools create $IDAT_PED_BED_MERGE_NODE_POOL \
  --cluster=$CLUSTER_NAME \
  --workload-metadata=GKE_METADATA \
  --machine-type=$VM \
  --zone=$ZONE \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=5 
  # --num-nodes=1

echo updating the autoscaling profile for the $IDAT_PED_BED_MERGE_NODE_POOL to optimize utilization
gcloud container clusters update $CLUSTER_NAME \
    --autoscaling-profile=optimize-utilization \
    --node-pool=$IDAT_PED_BED_MERGE_NODE_POOL \
    --region=$ZONE

# gcloud container node-pools update POOL_NAME \
#     --cluster CLUSTER_NAME \
#     --machine-type e2-highmem-16 \
#     --zone ZONE  # Or --region REGION for regional clusters
