#global resources for Genotools Server
ZONE: europe-west4-a
CLUSTER_NAME: gtserver-eu-west4
PROJECT_ID: GOOGLE_CLOUD_PROJECT
nodePools:
  appsNodePool: apps-node-pool
  genotoolsApiNodePool: genotools-api-node
  gtPrecheckApiNodePool: gtprecheck-api-node

persistentVOLUME: gtserver-pv
persistentVOLUMECLAIM: gtserver-pvc
GCP_SA: gsa-gtserver
#kubernetes resources
ksaBucket: ksa-bucket-access


namespace: kns-gtserver

dnsName: genotools-server.com
ingressName: gtserver-ingress
globalIPName: gke-ingress

gcpSecretManager:
  ksaExternalSecret: ksa-external-secrets
  secretStoreName: gcp-secrets-store
  externalSecretName: gcp-external-secret
  kubernetesSecretName: gtserver-secrets


pgSQL:
  ksaName: ksa-postgresql
  secrets:
    secName: gke-cloud-sql-secrets
    dbName: database
    user: username
    pass: password

sslRelated:
  caIssuer: letsencrypt-staging
  sec: gtserver-ssl

appservices:
  gp2browserApp:
    image: europe-west4-docker.pkg.dev/GOOGLE_CLOUD_PROJECT/gt-server/gp2-browser-app:COMMIT_SHA
    svcName: gp2-browser-app-svc
    iapBackend: gp2-browser-iap-backend
    iapSec: gp2-browser-iap-sec
    podName: gp2-browser-app-pod
    mountPath: /app/data

  genotrackerApp:
    image: europe-west4-docker.pkg.dev/GOOGLE_CLOUD_PROJECT/gt-server/genotracker-app:COMMIT_SHA 
    svcName: genotracker-app-svc
    iapBackend: genotracker-app-iap-backend
    iapSec: genotracker-app-iap-sec
    podName: genotracker-app-pod
    ksaPGSQL: ksa-postgresql
microservices:
  genotrackerApi:
    image: europe-west4-docker.pkg.dev/GOOGLE_CLOUD_PROJECT/gt-server/genotracker-api:COMMIT_SHA 
    svcName: genotracker-api-svc
    podName: genotracker-api-pod
    apiKey:
      name: genotracker-secret
      key: GENOTRACKER_API_KEY
  genotoolsApi:
    image: europe-west4-docker.pkg.dev/GOOGLE_CLOUD_PROJECT/gt-server/genotools-api:COMMIT_SHA 
    svcName: genotools-api-svc
    podName: genotools-api-pod
    mountPath: /app/genotools_api/data
    apiKey:
      name: genotools-api-sec
      key: GENOTOOLS_API_KEY #genotools-api-key
    patKey:
      name: genotools-api-sec
      key: EMAIL_PAT #genotools-api-key

  gtPrecheckAPI:
    image: europe-west4-docker.pkg.dev/GOOGLE_CLOUD_PROJECT/gt-server/gtprecheck-api:COMMIT_SHA 
    svcName: gt-precheck-api-svc
    podName: gt-precheck-api-pod
    mountPath: /app/data
