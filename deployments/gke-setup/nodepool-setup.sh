echo Creating nodepool, Will take a while...
bash nodepools/add-apps-node-pool.sh
bash nodepools/add-genotools-api-node-pool.sh
bash nodepools/add-gtprecheck-node-pool.sh

echo All Done, Please remember to delete the GCP resources manually.
echo "You can check the node pools with: gcloud container node-pools list --cluster=$CLUSTER_NAME --zone=$ZONE"
echo "You can check the nodes with: kubectl get nodes -o wide"
