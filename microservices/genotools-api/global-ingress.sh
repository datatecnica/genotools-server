export K8S_NAMESPACE=kns-gtserver #gke-ns-gtserver-eu-west4
export GENOTOOLS_API_SVC_NODEPORT=genotools-api-svc-nodeport
export SERVER_INGRESS=gtserver-ingress
export KSA=ksa-bucket-access #ksa
# export DNS_Release=genotoolsserver.com
export DNS_Test=genotools-server.com


cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: Service
metadata:
  name: ${GENOTOOLS_API_SVC_NODEPORT}
  namespace: ${K8S_NAMESPACE}
spec:
  type: NodePort #LoadBalancer
  selector:
    app: genotools-api-pod
  ports:
    - port: 8000
      targetPort: 8080
EOF
# cat <<EOF | kubectl apply -f -
# ---
# apiVersion: v1
# kind: Service
# metadata:
#   name: ${GP2_BROWSER_SVC_NODE_PORT}
#   namespace: ${K8S_NAMESPACE}
# spec:
#   type: NodePort
#   selector:
#     app: ${GP2_BROWSER_POD_NAME}
#   ports:
#     - protocol: TCP
#       port: 8000
#       targetPort: 8080
# EOF
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
    kubernetes.io/ingress.global-static-ip-name: "gke-ingress"  
spec:
  rules:
  # - host: gp2-browser.${DNS_Test}
  #   http:
  #     paths:
  #     - path: /
  #       pathType: Prefix
  #       backend:
  #         service:
  #           name: ${GP2_BROWSER_SVC_NODE_PORT}
  #           port: 
  #             number: 8000
  - host: genotools-api.${DNS_Test}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ${GENOTOOLS_API_SVC_NODEPORT}
            port: 
              number: 8000
      - path: /run-genotools/
        pathType: Prefix
        backend:
          service:
            name: ${GENOTOOLS_API_SVC_NODEPORT}
            port: 
              number: 8000                            
EOF
