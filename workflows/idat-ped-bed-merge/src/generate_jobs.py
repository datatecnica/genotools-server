# generate_configs.py
from jinja2 import Environment, FileSystemLoader

def generate_idat_ped_job_files(batch_folder_path, exec_folder_path, study, k8s_namespace, pv_claim, service_account_name, gke_nodepools):#, user_email):

    # Set up Jinja2 environment to load templates from the current directory
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('idat_ped.jinja2')

    # Load data from the YAML file
    with open(str(batch_folder_path)+f'/all_idat_ped_{study.lower()}_scripts.txt') as f:
        scripts = f.readlines()

    # Generate a YAML file for each device
    counter=1
    params = {}
    for script in scripts:
        output_filename = f"{str(batch_folder_path)}/job_idat_ped_{study}_{counter}.yaml"
        params["name"] = script.split("/")[-1].split(".")[0]
        params["script"] = script.split("/")[-1] #script
        params["exec_folder_path"] = exec_folder_path
        # params["batch_folder_path"] = batch_folder_path.split("/")[2]+"/"+"/".join(batch_folder_path.split("/")[3:])
        params["batch_folder_path"] = batch_folder_path
        params["k8s_namespace"] = k8s_namespace
        params["pv_claim"] = pv_claim
        params["service_account_name"] = service_account_name
        params["gke_nodepools"] = gke_nodepools
        #params["user_email"] = user_email


        rendered_yaml = template.render(params=params)
        cleaned_output = rendered_yaml.replace("\\", "")

        with open(output_filename, 'w') as f:
            f.write(cleaned_output)
        print(f"Generated {output_filename}")
        counter+=1

def generate_ped_bed_job_files(batch_folder_path, exec_folder_path, k8s_namespace, pv_claim, service_account_name, gke_nodepools):#, user_email):

    # Set up Jinja2 environment to load templates from the current directory
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('ped_bed.jinja2')

    # Load data from the YAML file
    with open(str(batch_folder_path)+'/all_ped_bed_scripts.txt') as f:
        scripts = f.readlines()

    # Generate a YAML file for each device
    counter=1
    params = {}
    for script in scripts:
        output_filename = f"{str(batch_folder_path)}/job_ped_bed_{counter}.yaml"
        params["name"] = script.split("/")[-1].split(".")[0]
        params["script"] = script.split("/")[-1] #script
        params["exec_folder_path"] = exec_folder_path
        # params["batch_folder_path"] = batch_folder_path
        # params["batch_folder_path"] = "gs://"+exec_folder_path.split("/")[2]+"/"+"/".join(batch_folder_path.split("/")[3:])
        params["batch_folder_path"] = batch_folder_path
        params["k8s_namespace"] = k8s_namespace
        params["pv_claim"] = pv_claim
        params["service_account_name"] = service_account_name
        params["gke_nodepools"] = gke_nodepools
        #params["user_email"] = user_email
        
        rendered_yaml = template.render(params=params)
        cleaned_output = rendered_yaml.replace("\\", "")

        with open(output_filename, 'w') as f:
            f.write(cleaned_output)
        print(f"Generated {output_filename}")
        counter+=1                

def generate_merge_beds_job_files(batch_folder_path, exec_folder_path, k8s_namespace, pv_claim, service_account_name, gke_nodepools): #, user_email):

    # Set up Jinja2 environment to load templates from the current directory
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('merge_beds.jinja2')

    # Load data from the YAML file
    with open(str(batch_folder_path)+'/all_bed_merge_scripts.txt') as f:
        scripts = f.readlines()

    # Generate a YAML file for each device
    counter=1
    params = {}
    for script in scripts:
        output_filename = f"{str(batch_folder_path)}/job_merge_bed_{counter}.yaml"
        params["name"] = script.split("/")[-1].split(".")[0]
        params["script"] = script.split("/")[-1] #script
        params["exec_folder_path"] = exec_folder_path
        # params["batch_folder_path"] = batch_folder_path
        # params["batch_folder_path"] = "gs://"+exec_folder_path.split("/")[2]+"/"+"/".join(batch_folder_path.split("/")[3:])
        params["batch_folder_path"] = batch_folder_path
        params["k8s_namespace"] = k8s_namespace
        params["pv_claim"] = pv_claim
        params["service_account_name"] = service_account_name
        params["gke_nodepools"] = gke_nodepools
        #params["user_email"] = user_email

        rendered_yaml = template.render(params=params)
        cleaned_output = rendered_yaml.replace("\\", "")

        with open(output_filename, 'w') as f:
            f.write(cleaned_output)
        print(f"Generated {output_filename}")
        counter+=1                        