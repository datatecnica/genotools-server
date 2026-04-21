echo  -e "\033[31mNow deploying Helm Charts for external secret operator, it may take a while...\033[0m"
bash all/secrets-operator.sh



echo   -e "\033[31mNow creating namespace "argocd" for argo cd and installing argocd Helm Charts, it may take a while...\033[0m"

kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml --wait


echo   -e "\033[31mNow creating namespace "argo" for argo workflow, it may take a while...\033[0m"
kubectl create namespace argo
echo  -e "\033[31m Now deploying Helm Charts for argo workflow, it may take a while...\033[0m"
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.7.4/install.yaml --wait

echo   -e "\033[31mNow starting setup for argo rollout, it may take a while...\033[0m"

helm repo add argo https://argoproj.github.io/argo-helm

helm install argo-rollouts argo/argo-rollouts \
  -n argo-rollouts \
  --create-namespace --wait


echo -e "\033[31mAll done, You can now deploy genotools-server helm charts from ../../dep/helm/dev, staging or prod directories.\033[0m"

echo -e "\033[31mAll done, You can now deploy genotools-server helm charts from ../../dep/helm/dev, staging or prod directories.\033[0m"


