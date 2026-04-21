apiVersion: external-secrets.io/v1
kind: ClusterSecretStore
metadata:
  name: {{ .Values.gcpSecretManager.secretStoreName }}
spec:
  provider:
    gcpsm:
      auth:
        workloadIdentity:
          clusterLocation: {{ .Values.ZONE }}
          clusterName: {{ .Values.CLUSTER_NAME }} 
          serviceAccountRef:
            name: {{ .Values.gcpSecretManager.ksaExternalSecret }} 
      projectID: {{ .Values.PROJECT_ID }}
