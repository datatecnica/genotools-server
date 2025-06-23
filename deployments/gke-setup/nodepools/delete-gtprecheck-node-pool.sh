export ZONE=europe-west4-a
export GCP_PROJECT=gp2-release-terra
export CLUSTER_NAME=gtcluster-eu-west4
export GT_PRECHECK_NODE_POOL=gtprecheck-api-node
export VM=c4-highmem-16

echo setting gcp project: $GCP_PROJECT

gcloud config set project $GCP_PROJECT

echo Deleting node-pool: $GT_PRECHECK_NODE_POOL from $CLUSTER_NAME k8s cluster in zone: $ZONE

gcloud container node-pools delete $GT_PRECHECK_NODE_POOL \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE