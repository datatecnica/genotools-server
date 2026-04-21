# This script creates K8S and sets up a persistent volume and claim for a GCP bucket in a GKE cluster.
#!/bin/bash
ZONE=europe-west4-a
REGION=europe-west4
PROJECT_ID=gp2-release-terra
CLUSTER_NAME=gke-prod-cluster

BUCKET_NAME_IDATS=gp2_idats
# BUCKET_NAME_APPS=genotools-server
PV=gtserver-pv-idats
PVC=gtserver-pvc-idats
#alsof for BUCKET_NAME_APPS
# PV_APPS=gtserver-pv-apps
# PVC_APPS=gtserver-pvc-apps

#service accounts
#GCP SA
GCP_SA=gsa-prod-gtserver
KSA_ExternalSecret_pregenotools=ksa-external-secrets
#####


K8S_NAMESPACE_PREGENOTOOLS=kns-pregenotools

KSA_Bucket_idats=ksa-bucket-idats

echo setting gcp project: $PROJECT_ID

gcloud config set project $PROJECT_ID

echo -e "\033[32mRetrieving Auth Credentials for the created Cluster: $CLUSTER_NAME, in zone: $ZONE\033[0m"

gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --dns-endpoint

echo enabling gateway-api add-on for the cluster
gcloud container clusters update $CLUSTER_NAME \
    --zone $ZONE \
    --gateway-api=standard
# gcloud container clusters describe $CLUSTER_NAME |grep gatewayApiConfig
echo enabling HttpLoadBalancing add-on for ingress

gcloud container clusters update $CLUSTER_NAME --update-addons=HttpLoadBalancing=ENABLED --zone $ZONE

# gcloud container clusters describe $CLUSTER_NAME |grep -C 5 httpload

echo -e "\033[32mDescribing the created cluster: $CLUSTER_NAME in zone: $ZONE\033[0m"

gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE



echo -e "\033[32mCreating k8s namespace: $K8S_NAMESPACE_PREGENOTOOLS, to better manage cluster resources in case we have multiple clusters\033[0m"

kubectl create namespace $K8S_NAMESPACE_PREGENOTOOLS


echo -e "\033[32mset current name space $K8S_NAMESPACE_PREGENOTOOLS for the cluster\033[0m"

kubectl config set-context --current --namespace=$K8S_NAMESPACE_PREGENOTOOLS

echo -e "\033[32mSetting up a cloud storage bucket: $BUCKET_NAME_IDATS, PLEASE IGNORE IF bucket is already there\033[0m"

gcloud storage buckets create gs://$BUCKET_NAME_IDATS \
  --location $REGION --uniform-bucket-level-access --project $PROJECT_ID


echo -e "\033[32mSetting up IAM policy for the bucket: $BUCKET_NAME_IDATS and allow access via: $GCP_SA service account we created above\033[0m"

# gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME_IDATS \
#   --member "serviceAccount:$GCP_SA@$PROJECT_ID.iam.gserviceaccount.com" \
#   --role "roles/storage.objectAdmin"


gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME_IDATS \
  --member "serviceAccount:$GCP_SA@$PROJECT_ID.iam.gserviceaccount.com" \
  --role "roles/storage.objectViewer"


######### BUCKET_NAME_APPS Block ##########
# echo -e "\033[32mSetting up a cloud storage bucket: $BUCKET_NAME_APPS, PLEASE IGNORE IF bucket is already there\033[0m"

# gcloud storage buckets create gs://$BUCKET_NAME_APPS \
#   --location $REGION --uniform-bucket-level-access --project $PROJECT_ID


# echo -e "\033[32mSetting up IAM policy for the bucket: $BUCKET_NAME_APPS and allow access via: $GCP_SA service account we created above\033[0m"

# gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME_APPS \
#   --member "serviceAccount:$GCP_SA@$PROJECT_ID.iam.gserviceaccount.com" \
#   --role "roles/storage.objectAdmin"

######### END BUCKET_NAME_APPS Block ######


######
echo -e "\033[32mNow creating k8s service account: $KSA_Bucket_idats in namespace: $K8S_NAMESPACE_PREGENOTOOLS to access the bucket: $BUCKET_NAME_IDATS \033[0m"

kubectl create serviceaccount $KSA_Bucket_idats --namespace $K8S_NAMESPACE_PREGENOTOOLS
# kubectl get serviceaccount ${KSA_Bucket} --namespace ${K8S_NAMESPACE}


echo -e "\033[32mNow binding k8s sa: $KSA_Bucket_idats to impersonate as real gcp service account via $GCP_SA for bucket access\033[0m"

gcloud iam service-accounts add-iam-policy-binding $GCP_SA@$PROJECT_ID.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:$PROJECT_ID.svc.id.goog[${K8S_NAMESPACE_PREGENOTOOLS}/${KSA_Bucket_idats}]" \


echo -e "\033[32mAnnotating k8s sa: $KSA_Bucket_idats to use the gcp sa: $GCP_SA for bucket access\033[0m"
kubectl annotate serviceaccount $KSA_Bucket_idats \
    --namespace ${K8S_NAMESPACE_PREGENOTOOLS} \
    iam.gke.io/gcp-service-account=$GCP_SA@${PROJECT_ID}.iam.gserviceaccount.com



echo -e "\033[32mCreating Cluster Level persistent volume: $PV\033[0m"

cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ${PV}
spec:
  accessModes:
  - ReadOnlyMany
  capacity:
    storage: 500Gi
  storageClassName: gtserver-pv1
  mountOptions:
    - implicit-dirs
    - ro
  csi:
    driver: gcsfuse.csi.storage.gke.io
    volumeHandle: ${BUCKET_NAME_IDATS}
EOF

# echo -e "\033[32mCreating Cluster Level persistent volume: $PV_APPS\033[0m"

# cat <<EOF | kubectl apply -f -
# ---
# apiVersion: v1
# kind: PersistentVolume
# metadata:
#   name: ${PV_APPS}
# spec:
#   accessModes:
#   - ReadWriteMany
#   capacity:
#     storage: 500Gi
#   storageClassName: gtserver-pv-apps
#   mountOptions:
#     - implicit-dirs
#   csi:
#     driver: gcsfuse.csi.storage.gke.io
#     volumeHandle: ${BUCKET_NAME_APPS}
# EOF


echo -e "\033[32mAlso creating persistent volume claim $PVC to use $PV in the namespace: $K8S_NAMESPACE_PREGENOTOOLS\033[0m"
cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC}
  namespace: ${K8S_NAMESPACE_PREGENOTOOLS}
spec:
  accessModes:
  - ReadOnlyMany
  resources:
    requests:
      storage: 500Gi
  volumeName: ${PV}
  storageClassName: gtserver-pv1
EOF

# echo -e "\033[32mAlso creating persistent volume claim $PVC_APPS to use $PV_APPS in the namespace: $K8S_NAMESPACE\033[0m"
# cat <<EOF | kubectl apply -f -
# ---
# apiVersion: v1
# kind: PersistentVolumeClaim
# metadata:
#   name: ${PVC_APPS}
#   namespace: ${K8S_NAMESPACE}
# spec:
#   accessModes:
#   - ReadWriteMany
#   resources:
#     requests:
#       storage: 500Gi
#   volumeName: ${PV_APPS}
#   storageClassName: gtserver-pv-apps
# EOF

#Also create KSA for secret manager access
echo -e "\033[32mCreate a KSA $KSA_ExternalSecret_pregenotools and configure to have access to Secret Manage by binding it to the GCP SA using Workload Identity Federation for GKE.\033[0m"

cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${KSA_ExternalSecret_pregenotools}
  namespace: ${K8S_NAMESPACE_PREGENOTOOLS}
  annotations:
    iam.gke.io/gcp-service-account: ${GCP_SA}@${PROJECT_ID}.iam.gserviceaccount.com
EOF
echo -e "\033[32mBinding the k8s sa: $KSA_ExternalSecret_pregenotools to impersonate as real gcp service account via $GCP_SA for secret manager access\033[0m"
gcloud iam service-accounts add-iam-policy-binding $GCP_SA@$PROJECT_ID.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:$PROJECT_ID.svc.id.goog[$K8S_NAMESPACE_PREGENOTOOLS/$KSA_ExternalSecret_pregenotools]" 
