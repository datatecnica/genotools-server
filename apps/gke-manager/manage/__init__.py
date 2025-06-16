import os
from google.cloud import secretmanager
import json
import pandas as pd

client = secretmanager.SecretManagerServiceClient()


# secret_name = "projects/664722061460/secrets/gke-manager-sec/versions/1"
secret_name = "projects/776926281950/secrets/gke-manager-sec/versions/1"


response = client.access_secret_version(request={"name": secret_name})
secret_value_json = response.payload.data.decode("UTF-8")

# Load the JSON string into a dictionary
secret_value_dict = json.loads(secret_value_json)

os.environ["CLIENT_ID"] = secret_value_dict['Oauth_CLIENT_ID']
os.environ["CLIENT_SECRET"] = secret_value_dict['Oauth_CLIENT_SECRET']
os.environ["REDIRECT_URI"] = secret_value_dict['Oauth_REDIRECT_URI']

