echo Deploying helm charts, It may take a while...
echo Now deploying Helm Charts for external secret operator, it may take a while...
bash packages/secrets-operator.sh
echo Done deploying Helm Charts for external secret operator.
echo Now deploying Helm Charts for cert-manager, it may take a while...
bash packages/cert-manager.sh
echo All done, You can now deploy genotools-server helm charts from dev/staging/prod directories.