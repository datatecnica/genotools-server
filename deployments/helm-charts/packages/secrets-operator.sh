export ZONE=europe-west4-a
export CLUSTER_NAME=gtserver-eu-west4
export GCP_SA=gsa-gtserver
export PROJECT_ID=gp2-code-test-env

echo retrieving Auth Credentials for the created Cluster: $CLUSTER_NAME, in zone: $ZONE

gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE


echo add eso helm repo
helm repo add external-secrets https://charts.external-secrets.io
helm repo update

echo installing the external-secrets repository using Helm

helm install external-secrets \
   external-secrets/external-secrets \
    -n external-secrets \
    --create-namespace \
    --wait
    # --set installCRDs=true \
    
# THIS BELOW WORKED - Testing above
# helm upgrade -install external-secrets external-secrets/external-secrets \
#     --set 'serviceAccount.annotations.iam\.gke\.io\/gcp-service-account'="$GCP_SA@$PROJECT_ID.iam.gserviceaccount.com" \
#     --namespace external-secrets \
#     --create-namespace \
#     --debug \
#     --wait


kubectl get all -n external-secrets

# helm delete external-secrets --namespace external-secrets
# kubectl get SecretStores,ClusterSecretStores,ExternalSecrets --all-namespaces
