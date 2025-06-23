export ZONE=europe-west4-a
export CLUSTER_NAME=gtserver-eu-west4


# echo set default namespace for the cluster

# kubectl config set-context --current --namespace=default

echo retrieving Auth Credentials for the created Cluster: $CLUSTER_NAME, in zone: $ZONE

gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE


echo  Installing Cert Manager in the cert-manager namespace

kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.17.2/cert-manager.yaml


kubectl get pods --namespace cert-manager

kubectl get all --namespace cert-manager

kubectl explain Certificate
kubectl explain CertificateRequest
kubectl explain Issuer