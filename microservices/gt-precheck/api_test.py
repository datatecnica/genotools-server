import requests
# import json
# from pprint import pprint


# payload = {
#     "user_email": "syed@datatecnica.com",
#     "geno_path": "/app/data/syed-test/input",
#     "out_path": "/app/data/gt_precheck/output/",
#     "key_path": "/app/data/gt_precheck",
#     "ref_panel": "/app/data/ref/new_panel/ref_panel_gp2_prune_rm_underperform_pos_update",
#     "ref_labels": "/app/data/ref/new_panel/ref_panel_ancestry_updated.txt",
#     "model": "/app/data/ref/models/python3_11/GP2_merge_release6_NOVEMBER_ready_genotools_qc_umap_linearsvc_ancestry_model.pkl"
# }


# payload = {
#     "user_email": "syed@datatecnica.com",
#     "geno_path": "/app/data/input",
#     "out_path": "/app/data/out/",
#     "key_path": "/app/data",
#     "ref_panel": "/app/data/ref/new_panel/ref_panel_gp2_prune_rm_underperform_pos_update",
#     "ref_labels": "/app/data/ref/new_panel/ref_panel_ancestry_updated.txt",
#     "model": "/app/data/ref/models/python3_11/GP2_merge_release6_NOVEMBER_ready_genotools_qc_umap_linearsvc_ancestry_model.pkl"
# }



payload = {
    "user_email": "syed@datatecnica.com",
    "geno_path": "/app/data/syed-test/gt_precheck/input/sub1",
    "out_path": "/app/data/syed-test/gt_precheck/output4/",
    "key_path": "/app/data/syed-test/gt_precheck",
    "ref_panel": "/app/data/ref/new_panel/ref_panel_gp2_prune_rm_underperform_pos_update",
    "ref_labels": "/app/data/ref/new_panel/ref_panel_ancestry_updated.txt",
    "model": "/app/data/ref/models/python3_11/GP2_merge_release6_NOVEMBER_ready_genotools_qc_umap_linearsvc_ancestry_model.pkl"
}


# headers = {
#     "X-API-Key": api_key
# }
# url = "https://syed-test-776926281950.europe-west4.run.app/process_carriers"
# response = requests.get("http://127.0.0.1:8001/")
# response = requests.get("http://gt-precheck.genotools-server.com")
# response = requests.get("https://gt-precheck-664722061460.europe-west1.run.app/health")
# response = requests.get("https://gt-precheck-776926281950.europe-west1.run.app/health")
# response = requests.get("http://gt-precheck.genotools-server.com")
response = requests.get("https://gt-precheck.genotools-server.com")
print(f"Health check: {response.json()}")

# for local testing
# url = 'http://gt-precheck.genotools-server.com/prechecks'
url = 'https://gt-precheck.genotools-server.com/prechecks'
# url = 'http://127.0.0.1:8001/prechecks' 
# url = 'http://gt-precheck.genotools-server.com/prechecks'
# url = 'https://gt-precheck-664722061460.europe-west1.run.app/prechecks' 
# url = 'https://gt-precheck-776926281950.europe-west1.run.app/prechecks'

try:
    response = requests.post(
        url,
        json=payload
        # headers=headers
    )
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    response = None