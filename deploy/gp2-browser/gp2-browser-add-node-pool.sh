export ZONE=europe-west4-a
export GCP_PROJECT=gp2-code-test-env
export CLUSTER_NAME=gtserver-eu-west4
export GP2_BROWSER_APP_NODE_POOL=gp2-browser-app-node-pool
export VM=e2-standard-4

echo setting gcp project: $GCP_PROJECT

gcloud config set project $GCP_PROJECT

echo Adding node-pool: $GP2_BROWSER_APP_NODE_POOL to use with the gp2_browser app deployment on $CLUSTER_NAME, k8s cluster in zone: $ZONE -  Please change machine type if needed

gcloud container node-pools create $GP2_BROWSER_APP_NODE_POOL \
  --cluster=$CLUSTER_NAME \
  --machine-type=$VM \
  --zone=$ZONE \
  --num-nodes=1

