PROJECT_ID=gp2-release-terra
MANAGED_ZONE_NAME=gtserver
LOCATION=global
DOMAIN_NAME=genotoolsserver.com
DNS_AUTH_NAME=prod-dns-auth
DNS_AUTH_CERT=prod-dns-cert
DNS_AUTH_CERT_MAP=prod-dns-cert-map

echo -e  "\033[32mSetting gcp project: $PROJECT_ID\033[0m"
gcloud config set project $PROJECT_ID


echo -e "\033[32mDescribing the created certificate: $DNS_AUTH_CERT to check its status:\033[0m"
gcloud certificate-manager certificates describe $DNS_AUTH_CERT

echo -e "\033[32mCreating Certificate Map: $DNS_AUTH_CERT_MAP to map the certificate to domain names:\033[0m"
gcloud certificate-manager maps create $DNS_AUTH_CERT_MAP

echo -e "\033[32mListing all the certificate maps to verify the created map:\033[0m"
gcloud certificate-manager maps list

echo -e "\033[32mCreating Certificate Map Entries $DNS_AUTH_CERT-0 and $DNS_AUTH_CERT-1 to map the certificate to domain names:\033[0m"
gcloud certificate-manager maps entries create $DNS_AUTH_CERT-0 --map=$DNS_AUTH_CERT_MAP --certificates=$DNS_AUTH_CERT --hostname="${DOMAIN_NAME}"
gcloud certificate-manager maps entries create $DNS_AUTH_CERT-1 --map=$DNS_AUTH_CERT_MAP --certificates=$DNS_AUTH_CERT --hostname="*.${DOMAIN_NAME}"

gcloud certificate-manager maps entries list --map=$DNS_AUTH_CERT_MAP