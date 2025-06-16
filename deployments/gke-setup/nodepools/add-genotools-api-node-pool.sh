export ZONE=europe-west4-a
# export PROJECT_ID=gp2-release-terra
export PROJECT_ID=gp2-code-test-env
export CLUSTER_NAME=gtserver-eu-west4
export GENOTOOLS_API_NODE=genotools-api-node
export VM=c4-highmem-32

echo setting gcp project: $PROJECT_ID

gcloud config set project $PROJECT_ID

echo retrieving Auth Credentials for the created Cluster: $CLUSTER_NAME, in zone: $ZONE

gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE


echo Adding node-pool: $GENOTOOLS_API_NODE to use with the genotools-api service on $CLUSTER_NAME k8s cluster in zone: $ZONE -  Please change machine type if needed

gcloud container node-pools create $GENOTOOLS_API_NODE \
  --cluster=$CLUSTER_NAME \
  --machine-type=$VM \
  --zone=$ZONE \
  --num-nodes=1
