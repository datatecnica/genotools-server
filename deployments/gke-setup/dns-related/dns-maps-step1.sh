PROJECT_ID=gp2-release-terra
MANAGED_ZONE_NAME=gtserver
LOCATION=global
DOMAIN_NAME=genotoolsserver.com
DNS_AUTH_NAME=prod-dns-auth
DNS_AUTH_CERT=prod-dns-cert
DNS_AUTH_CERT_MAP=prod-dns-cert-map

echo -e  "\033[32mSetting gcp project: $PROJECT_ID\033[0m"
gcloud config set project $PROJECT_ID

echo -e  "\033[32mCreating Managed Zone: $MANAGED_ZONE_NAME for DNS: $DOMAIN_NAME\033[0m"
gcloud dns managed-zones create $DOMAIN_NAME  \
    --description "Public Zone for domain $DOMAIN_NAME"  \
    --dns-name $DOMAIN_NAME \
    --visibility public        


echo -e "\033[32mCreating DNS Authorization for domain: genotoolsserver.com\033[0m"
gcloud certificate-manager dns-authorizations create $DNS_AUTH_NAME --domain=$DOMAIN_NAME --location=$LOCATION

echo -e "\033[32mDescribing DNS Authorization to get the DNS TXT record to be created in the domain registrar\033[0m"
gcloud certificate-manager dns-authorizations describe $DNS_AUTH_NAME --location=$LOCATION


CNAME_DATA=$(gcloud certificate-manager dns-authorizations describe $DNS_AUTH_NAME \
    --location=$LOCATION --format="value(dnsResourceRecord.data)")

CNAME_NAME=$(gcloud certificate-manager dns-authorizations describe $DNS_AUTH_NAME \
    --location=$LOCATION --format="value(dnsResourceRecord.name)")


echo -e "\033[32mCreating the DNS CNAME: $CNAME_NAME record and data: $CNAME_DATA in $MANAGED_ZONE_NAME to verify domain ownership:\033[0m"

gcloud dns record-sets transaction start --zone=$MANAGED_ZONE_NAME
gcloud dns record-sets transaction add "${CNAME_DATA}" \
    --name="${CNAME_NAME}" \
    --ttl=300 \
    --type=CNAME \
    --zone=$MANAGED_ZONE_NAME
gcloud dns record-sets transaction execute --zone=$MANAGED_ZONE_NAME

echo -e "\033[32mVerifying the created CNAME record via dig command:\033[0m"
dig +short -t CNAME $CNAME_NAME

echo -e "\033[32mCreating Managed Certificate for domain: genotoolsserver.com\033[0m"
echo -e "\033[31mPlease note that this step takes long time, so please verify that Certificates are active before using them.\033[0m"
gcloud certificate-manager certificates create $DNS_AUTH_CERT --domains="*.$DOMAIN_NAME,$DOMAIN_NAME" --dns-authorizations=$DNS_AUTH_NAME

echo -e "\033[32mListing all the certificates to verify the created certificate:\033[0m"
gcloud certificate-manager certificates list

echo -e "\033[32mDescribing the created certificate: $DNS_AUTH_CERT to check its status:\033[0m"
gcloud certificate-manager certificates describe $DNS_AUTH_CERT

echo -e "\033[31mPlease wait for $DNS_AUTH_CERT to be active and then run scritp dns-maps-step2\033[0m"

