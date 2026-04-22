ZONE=europe-west4-a
PROJECT_ID=gp2-release-terra
CLUSTER_NAME=gke-prod-cluster
APPS_NODE_POOL=apps-node
VM=e2-standard-8

echo setting gcp project: $PROJECT_ID

gcloud config set project $PROJECT_ID

echo Deleting node-pool: $APPS_NODE_POOL from $CLUSTER_NAME k8s cluster in zone: $ZONE

gcloud container node-pools delete $APPS_NODE_POOL \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE