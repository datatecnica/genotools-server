docker buildx build --platform linux/amd64 -t europe-west4-docker.pkg.dev/gp2-release-terra/genotools/genotools-api . --load

docker push europe-west4-docker.pkg.dev/gp2-release-terra/genotools/genotools-api

# gcloud run deploy genotools-api \
#   --image europe-west4-docker.pkg.dev/gp2-release-terra/genotools/genotools-api \
#   --platform managed \
#   --region europe-west4 \
#   --service-account genotools-server@gp2-release-terra.iam.gserviceaccount.com

gcloud run deploy genotools-api \
  --image europe-west4-docker.pkg.dev/gp2-release-terra/genotools/genotools-api \
  --platform managed \
  --region europe-west4 \
  --service-account genotools-server@gp2-release-terra.iam.gserviceaccount.com \
  --set-env-vars API_KEY="your_api_key_value",API_KEY_NAME="X-API-KEY" \
  --allow-unauthenticated






# setup api key
APIKEY=<APIKEY>
echo -n $APIKEY | gcloud secrets create genotools-api-key --data-file=-
echo -n $APIKEY | gcloud secrets versions add genotools-api-key --data-file=- 

# give service account access to key
gcloud secrets add-iam-policy-binding genotools-api-key \
    --member="serviceAccount:genotools-server@gp2-release-terra.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"