ZONE=europe-west4-a
REGION=europe-west4
CLUSTER_NAME=test-cluster
PROJECT_ID=gp2-testing-475115
VM=e2-standard-8
IDAT_PED_BED_MERGE_NODE_POOL=workflow-idat-ped-bed-nodepool


echo setting gcp project: $PROJECT_ID

gcloud config set project $PROJECT_ID

echo Deleting node-pool: $IDAT_PED_BED_MERGE_NODE_POOL from $CLUSTER_NAME k8s cluster in zone: $ZONE

gcloud container node-pools delete $IDAT_PED_BED_MERGE_NODE_POOL \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE