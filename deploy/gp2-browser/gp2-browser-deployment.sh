export K8S_NAMESPACE=gke-ns-gtserver-eu-west4
export GP2_BROWSER_APP_NODE_POOL=gp2-browser-app-node-pool
export GP2_BROWSER_POD_NAME=gp2-browser-pod
export GP2_BROWSER_SVC_NODE_PORT=gp2-browser-svc-nodeport
export PV=gtserver-pv
export PVC=gtserver-pvc
export SERVER_INGRESS=gtserver-ingress
export KSA=ksa


cat <<EOF | kubectl apply -f -
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: ${GP2_BROWSER_POD_NAME}
  name: ${GP2_BROWSER_POD_NAME}
  namespace: ${K8S_NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${GP2_BROWSER_POD_NAME}
  template:
    metadata:
      labels:
        app: ${GP2_BROWSER_POD_NAME}
      annotations:
        gke-gcsfuse/volumes: "true" 
    spec:
      serviceAccountName: ${KSA}
      volumes:
      - name: gcs-volume
        persistentVolumeClaim:
          claimName: ${PVC}
      containers:
      - image: us-east1-docker.pkg.dev/gp2-code-test-env/syed-test/genotools-server/apps/gp2-browser/gp2-browser:latest
        name: gp2-browser-container
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: gcs-volume
          mountPath: /app/data
      # Ensure pods are scheduled on nodes with the specified GCP VM type
      nodeSelector:
        cloud.google.com/gke-nodepool: ${GP2_BROWSER_APP_NODE_POOL}
EOF
cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: Service
metadata:
  name: ${GP2_BROWSER_SVC_NODE_PORT}
  namespace: ${K8S_NAMESPACE}
spec:
  type: NodePort
  selector:
    app: ${GP2_BROWSER_POD_NAME}
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8080
EOF
cat <<EOF | kubectl apply -f -
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${SERVER_INGRESS}
  namespace: ${K8S_NAMESPACE}
  annotations:
    kubernetes.io/ingress.class: "gce"
    kubernetes.io/ingress.allow-http: "true"
spec:
  rules:
  - http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: ${GP2_BROWSER_SVC_NODE_PORT}
            port: 
              number: 8000
EOF
