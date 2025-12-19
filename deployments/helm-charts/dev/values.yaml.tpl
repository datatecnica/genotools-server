#global resources for Genotools Server
ZONE: europe-west4-a
CLUSTER_NAME: GKE_CLUSTER_NAME #gke-test-cluster
PROJECT_ID: GOOGLE_CLOUD_PROJECT
nodePools:
  appsNodePool: apps-node
  genotoolsApiNodePool: genotools-api-node
  gtPrecheckApiNodePool: gtprecheck-api-node
  snpMatrixAPINodePool: snp-matrix-node-pool

persistentVOLUME: gtserver-pv
persistentVOLUMECLAIM: gtserver-pvc
GCP_SA: gsa-gtserver
#kubernetes resources
ksaBucket: ksa-bucket-access


namespace: kns-gtserver
# namespace: kns-test

dnsName: DNS_NAME #genotoolsservers.com #genotools-server.com
# ingressName: gtserver-ingress
# globalIPName: gke-ingress

argoRelated:
  argoRoles:
    roleName: argo-workflow-role
    bindingName: argo-workflow-role-binding
  argoWF:
    routeName: argo-route
    namespace: argo
    hostname: workflows.DNS_NAME
    serviceAccountName: argo-ksa
    workflowRoleBindingName: argo-workflow-role-binding
gatewayAPI:
  gatewayName: gt-server-gwapi
  certKey: gt-server-gateway
  globalIPName: STATIC_IP_NAME #gt-test-ip
#  routeName: genotools-routes

gcpSecretManager:
  ksaExternalSecret: ksa-external-secrets
  secretStoreName: gcp-secrets-store
  externalSecretName: gcp-external-secret
  kubernetesSecretName: gtserver-secrets
  iapRelated:
    apps:
      externalSecretName: apps-iap-secrets
      kubernetesSecretName: apps-iap-k8s-secret
      gcpSecretName: iap-related
    # genotrackerApp:
    #   externalSecretName: genotracker-app-iap-secret
    #   kubernetesSecretName: genotracker-app-iap-k8s-secret
    #   gcpSecretName: iap-related



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
  precisionMedApp:
    image: europe-west4-docker.pkg.dev/GOOGLE_CLOUD_PROJECT/APPS_REPO/precision-med-app:COMMIT_SHA
    containerName: precisionmed-app-cont
    svcName: precisionmed-app-svc
    iapPolicyBackend: precisionmed-app-iap-backend
    containerPort: 8080
    servicePort: 8000
    hostname: precision-med.DNS_NAME
    routeName: precisionmed-app-route
    podName: precisionmed-app-pod
    volume: precisionmed-app-vol
    mountPath: /app/data
    iapValues:
      ClientID: precisionmed_app_id
      ClientSecret: precisionmed_app_secret

  genotrackerApp:
    image: europe-west4-docker.pkg.dev/GOOGLE_CLOUD_PROJECT/APPS_REPO/genotracker-app:COMMIT_SHA
    containerName: genotracker-app-cont
    svcName: genotracker-app-svc
    iapPolicyBackend: genotracker-app-iap-backend
    containerPort: 8080
    servicePort: 8000
    hostname: genotracker-app.DNS_NAME
    routeName: genotracker-app-route
    podName: genotracker-app-pod
    volume: genotracker-app-vol
    mountPath: /app/data
    iapValues:
      ClientID: genotracker_app_id
      ClientSecret: genotracker_app_secret

    # ksaPGSQL: ksa-postgresql
    # iapBackend: genotracker-app-iap-backend
    # iapSec: genotracker-app-iap-sec

  gp2browserApp:
    image: europe-west4-docker.pkg.dev/GOOGLE_CLOUD_PROJECT/APPS_REPO/gp2-browser:COMMIT_SHA
    containerName: gp2browser-app-cont
    svcName: gp2browser-app-svc
    iapPolicyBackend: gp2browser-iap-backend
    containerPort: 8080
    servicePort: 8000
    hostname: gp2browser-app.DNS_NAME
    routeName: gp2browser-app-route
    podName: gp2browser-app-pod
    volume: gp2browser-app-vol
    mountPath: /app/data
    iapValues:
      ClientID: gp2browser_id
      ClientSecret: gp2browser_secret


microservices:
  genotrackerApi:
    image: europe-west4-docker.pkg.dev/GOOGLE_CLOUD_PROJECT/API_REPO/genotracker-api:COMMIT_SHA
    containerName: genotracker-api-cont
    svcName: genotracker-api-svc
    containerPort: 8080
    servicePort: 8000
    hostname: genotracker-api.DNS_NAME
    routeName: genotracker-api-route
    podName: genotracker-api-pod
    volume: genotracker-api-vol
    mountPath: /app/data
    apiKey:
      name: genotracker-secret
      key: GENOTRACKER_API_KEY
  genotoolsApi:
    image: europe-west4-docker.pkg.dev/GOOGLE_CLOUD_PROJECT/API_REPO/genotools-api:COMMIT_SHA
    containerName: genotools-api-cont
    svcName: genotools-api-svc
    containerPort: 8080
    servicePort: 8000
    # hostname: genotools-api.genotoolsserver.com
    hostname: genotools-api.DNS_NAME
    routeName: genotools-api-route
    podName: genotools-api-pod
    volume: genotools-api-vol
    mountPath: /app/genotools_api/data
    apiKey:
      name: genotools-api-sec
      key: GENOTOOLS_API_KEY #genotools-api-key
    patKey:
      name: genotools-api-sec
      key: EMAIL_PAT #genotools-api-key
