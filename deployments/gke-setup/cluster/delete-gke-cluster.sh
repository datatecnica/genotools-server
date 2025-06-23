export ZONE=europe-west4-a
export GCP_PROJECT=gp2-release-terra
export CLUSTER_NAME=gtcluster-eu-west4

echo setting gcp project: $GCP_PROJECT

gcloud config set project $GCP_PROJECT

echo Deleting node-pool: $CLUSTER_NAME k8s cluster in zone: $ZONE

gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE

