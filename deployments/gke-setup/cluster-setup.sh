echo Setting up cluster, It may take a while...
bash cluster/create-gke-cluster.sh
echo Setting necessary settings for the cluster including namespace, gcp service account, kubernetes service account and necessary authentication, persistent volume and persistent volume claim for the cluster resources
bash cluster/create-sa-ns-pv-pvc.sh

echo All Done, Please check the cluster status and the created resources
echo "You can check the cluster status with: kubectl get nodes -o wide"
echo "You can check the created resources with: kubectl get all -n kns-gtserver"
echo "You can check the created persistent volume with: kubectl get pv"
echo "You can check the created persistent volume claim with: kubectl get pvc -n kns-gtserver"
echo "You can check the created service accounts with: kubectl get serviceaccounts -n kns-gtserver"