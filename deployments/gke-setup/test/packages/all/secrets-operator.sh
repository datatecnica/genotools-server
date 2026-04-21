ZONE=europe-west4-a
CLUSTER_NAME=test-gke-prod-cluster
PROJECT_ID=gp2-release-terra


echo -e "\033[32mRetrieving Auth Credentials for the created Cluster: $CLUSTER_NAME, in zone: $ZONE\033[0m"

gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --dns-endpoint


echo  -e "\033[32mAdd helm repo\033[0m"
helm repo add external-secrets https://charts.external-secrets.io
helm repo update

echo  -e "\033[32mInstalling the external-secrets repository using Helm\033[0m"

helm install external-secrets \
   external-secrets/external-secrets \
    -n external-secrets \
    --create-namespace \
    --set installCRDs=true \
    --wait
    
    
echo  -e "\033[32mVerifying the external-secrets installation\033[0m"

kubectl get all -n external-secrets

# helm delete external-secrets --namespace external-secrets
# kubectl get SecretStores,ClusterSecretStores,ExternalSecrets --all-namespaces
