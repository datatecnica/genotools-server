#!/bin/bash

export ZONE=europe-west4-a
export ZONE1=europe-west4
export CLUSTER_NAME=gtserver-eu-west4
# export PROJECT_NUMBER=776926281950
# export PROJECT_ID=gp2-release-terra
export PROJECT_NUMBER=664722061460
export PROJECT_ID=gp2-code-test-env
export VM=e2-standard-8
#GCP SA
export GCP_SA=gsa-gtserver

echo Please make sure to replace/use correct GCP ProjectNumber: $PROJECT_NUMBER and ProjectID: $PROJECT_ID

echo setting gcp project: $PROJECT_ID

gcloud config set project $PROJECT_ID

echo Creating k8s cluster: $CLUSTER_NAME with VM: $VM machine-type in zone $ZONE with one worker node in gcp projetc: $PROJECT_ID. Also enable GcsFuseCsiDriver for gcp bucket access and secret manager

gcloud container clusters create $CLUSTER_NAME \
  --enable-secret-manager \
  --addons GcsFuseCsiDriver \
  --addons HttpLoadBalancing \
  --zone $ZONE \
  --workload-pool=$PROJECT_ID.svc.id.goog \
  --project $PROJECT_ID \
  --machine-type $VM \
  --num-nodes=1

echo Verify the Secret Manager add-on installation

gcloud container clusters describe $CLUSTER_NAME  --zone $ZONE | grep secretManagerConfig -A 1

echo verify that workload identity is enabled for the cluster
gcloud container clusters describe ${CLUSTER_NAME} --zone $ZONE | grep workloadPool

echo retrieving Auth Credentials for the created Cluster: $CLUSTER_NAME, in zone: $ZONE

gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE



echo Also create GCP service account: $GCP_SA to use with the k8s cluster. Please note that an Error saying SA exists is OK

gcloud iam service-accounts create $GCP_SA --project=$PROJECT_ID --display-name=$GCP_SA --description="GCP SA for GTServer"


echo enabling HttpLoadBalancing add-on for ingress

gcloud container clusters update $CLUSTER_NAME --update-addons=HttpLoadBalancing=ENABLED --zone $ZONE

echo enabling GcsFuseCsiDriver add-on for gcp bucket access, If failed, please check if the cluster is created with --addons GcsFuseCsiDriver and rerun this command
gcloud container clusters update $CLUSTER_NAME --update-addons=GcsFuseCsiDriver=ENABLED --zone $ZONE



#if pv or pvc can not be deleted
# kubectl patch pv sample-app-pv -p '{"metadata":{"finalizers":null}}' -n namespace1
