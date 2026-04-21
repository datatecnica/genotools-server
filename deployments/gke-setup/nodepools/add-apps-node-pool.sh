ZONE=europe-west4-a
PROJECT_ID=gp2-release-terra
CLUSTER_NAME=gke-prod-cluster
APPS_NODE_POOL=apps-node
VM=e2-standard-8

echo -e "\033[32mStarting to add node-pool: $APPS_NODE_POOL to use with the gp2_browser app deployment on $CLUSTER_NAME k8s cluster in zone: $ZONE -  Please change machine type if needed\033[0m"

echo setting gcp project: $PROJECT_ID

gcloud config set project $PROJECT_ID

echo retrieving Auth Credentials for the created Cluster: $CLUSTER_NAME, in zone: $ZONE

gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --dns-endpoint


echo Adding node-pool: $APPS_NODE_POOL to use with the gp2_browser app deployment on $CLUSTER_NAME k8s cluster in zone: $ZONE -  Please change machine type if needed


gcloud container node-pools create $APPS_NODE_POOL \
  --cluster=$CLUSTER_NAME \
  --machine-type=$VM \
  --zone=$ZONE \
  --num-nodes=1
  # --disk-size=$DISK_SIZE \
