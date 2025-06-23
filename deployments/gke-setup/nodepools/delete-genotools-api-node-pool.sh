export ZONE=europe-west4-a
export GCP_PROJECT=gp2-release-terra
export CLUSTER_NAME=gtcluster-eu-west4
# export APPS_NODE_POOL=apps-node-pool
export GENOTOOLS_API_NODE=genotools-api-node
export VM=c4-highmem-32

echo setting gcp project: $GCP_PROJECT

gcloud config set project $GCP_PROJECT

echo Deleting node-pool: $GENOTOOLS_API_NODE from $CLUSTER_NAME k8s cluster in zone: $ZONE

gcloud container node-pools delete $GENOTOOLS_API_NODE \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE