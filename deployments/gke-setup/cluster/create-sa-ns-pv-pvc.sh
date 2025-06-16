# This script creates K8S and sets up a persistent volume and claim for a GCP bucket in a GKE cluster.
#!/bin/bash
export ZONE=europe-west4-a
export ZONE1=europe-west4
export CLUSTER_NAME=gtserver-eu-west4

export PROJECT_NUMBER=664722061460
export PROJECT_ID=gp2-code-test-env

export BUCKET_NAME=gtserver-eu-west4-$PROJECT_ID
#service accounts
#GCP SA
export GCP_SA=gsa-gtserver


export K8S_NAMESPACE=kns-gtserver
export PV=gtserver-pv
export PVC=gtserver-pvc
export KSA_Bucket=ksa-bucket-access

export SECRET_VERSION=1


echo retrieving Auth Credentials for the created Cluster: $CLUSTER_NAME, in zone: $ZONE

gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE

# Removed (now in gke-cluster.sh ) as it should be created before changing - gcloud iam service-accounts create $GCP_SA --project=$PROJECT_ID


echo Creating k8s namespace: $K8S_NAMESPACE, to better manage cluster resources in case we have multiple clusters

kubectl create namespace $K8S_NAMESPACE


# echo set current name space $K8S_NAMESPACE for the cluster

# kubectl config set-context --current --namespace=$K8S_NAMESPACE

echo Setting up a cloud storage bucket: $BUCKET_NAME, PLEASE IGNORE IF bucket is already there

gcloud storage buckets create gs://$BUCKET_NAME \
  --location $ZONE1 --uniform-bucket-level-access --project $PROJECT_ID

echo Setting up IAM policy for the bucket: $BUCKET_NAME and allow access via: $GCP_SA service account we created above

gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME \
  --member "serviceAccount:$GCP_SA@$PROJECT_ID.iam.gserviceaccount.com" \
  --role "roles/storage.objectAdmin"



kubectl create serviceaccount $KSA_Bucket --namespace $K8S_NAMESPACE



# kubectl get serviceaccount ksa-bucket-access --namespace kns-gtserver -o yaml
# kubectl get serviceaccount ${KSA_NAME} --namespace ${NAMESPACE}



echo Now binding derived service account: $KSA_Bucket service to impersonate as real gcp service account via $GCP_SA for bucket access

gcloud iam service-accounts add-iam-policy-binding $GCP_SA@$PROJECT_ID.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:$PROJECT_ID.svc.id.goog[${K8S_NAMESPACE}/${KSA_Bucket}]" \

kubectl annotate serviceaccount $KSA_Bucket \
    --namespace ${K8S_NAMESPACE} \
    iam.gke.io/gcp-service-account=$GCP_SA@${PROJECT_ID}.iam.gserviceaccount.com


echo Creating Cluster Level persistent volume: $PV

cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ${PV}
spec:
  accessModes:
  - ReadWriteMany
  capacity:
    storage: 500Gi
  storageClassName: gtserver-pv1
  mountOptions:
    - implicit-dirs
    # - uid=1001
    # - gid=3003
  csi:
    driver: gcsfuse.csi.storage.gke.io
    volumeHandle: ${BUCKET_NAME}
    # volumeAttributes:
    #   skipCSIBucketAccessCheck: "true"
    #   gcsfuseMetadataPrefetchOnMount: "true"    
  # claimRef:
  #   name: ${PVC}
    # namespace: ${K8S_NAMESPACE}      
EOF
echo Also creating persistent volume claim $PVC to use $PV in the namespace: $K8S_NAMESPACE
cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC}
  namespace: ${K8S_NAMESPACE}
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  volumeName: ${PV}
  storageClassName: gtserver-pv1
EOF

echo Now for cloudsql related

export PostgreSQL_Instance=sql-genotools-server
export PGSQL_ROOT_PASS=syed@gtserver
# export DB_NAME=genotracker-db
# export GENOTRACKER_DB_USER=genotracker-user
# export GENOTRACKER_DB_USER_PASSWORD=gt_pass

export KSA_PGSQL=ksa-postgresql
# export KS_PGSQL_SEC=gke-cloud-sql-secrets
export KSA_ExternalSecret=ksa-external-secrets
#####

ROLES=("roles/cloudsql.admin" "roles/logging.logWriter" "roles/artifactregistry.reader", "roles/secretmanager.secretAccessor", "roles/iam.serviceAccountTokenCreator")

echo add the cloudsql.client, Log Writer and Artifact Registry Reader role to the GCP SA
for role in "${ROLES[@]}"; do
        gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$GCP_SA@$PROJECT_ID.iam.gserviceaccount.com" --role "$role"
    done


######
echo Create a KSA $KSA_PGSQL and configure to have access to Cloud SQL by binding it to the GCP SA using Workload Identity Federation for GKE.

cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${KSA_PGSQL}
  namespace: ${K8S_NAMESPACE}
  annotations:
    iam.gke.io/gcp-service-account: $GCP_SA@$PROJECT_ID.iam.gserviceaccount.com

EOF

echo Enable IAM binding of the GCP SA $GCP_SA to the KSA $KSA_PGSQL

gcloud iam service-accounts add-iam-policy-binding $GCP_SA@$PROJECT_ID.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:$PROJECT_ID.svc.id.goog[$K8S_NAMESPACE/$KSA_PGSQL]"


#Also create KSA for secret manager access
echo Create a KSA $KSA_ExternalSecret and configure to have access to Secret Manage by binding it to the GCP SA using Workload Identity Federation for GKE.

cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${KSA_ExternalSecret}
  namespace: ${K8S_NAMESPACE}
  annotations:
    iam.gke.io/gcp-service-account: ${GCP_SA}@${PROJECT_ID}.iam.gserviceaccount.com
EOF

gcloud iam service-accounts add-iam-policy-binding $GCP_SA@$PROJECT_ID.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:$PROJECT_ID.svc.id.goog[$K8S_NAMESPACE/$KSA_ExternalSecret]" 
    #--project=$PROJECT_ID
