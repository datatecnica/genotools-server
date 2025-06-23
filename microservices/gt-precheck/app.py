from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
import os
import pandas as pd
import json
from typing import  List, Tuple
from utils.cohorts import CohortsExtrator
# from processor.do_processing import Processor, execute_genotools
from genotools.utils import shell_do
import logging

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from datetime import datetime


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()


subject_submitted = "Job Submission Confirmation (gtserver: gt_precheck)"
subject_completed = "Job Completion Confirmation (gtserver: gt_precheck)"
body_submitted = 'You gt_precheck job has been submitted. You will receive an eamil upon job completion as well.'
body_completed = 'You gt_precheck job has been completed, Please check logs for details.'
sender_email = "si11080772@gmail.com"
pat = "qgetcdycqfkqpiaq"


def send_email(subject, body, sender_email, pat, recipient_email):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, pat)
    text = msg.as_string()
    server.sendmail(sender_email, recipient_email, text)
    server.quit()



def generate_report(cohort, out_path, key_path):
    try:
        df_pruned = pd.read_csv(out_path+cohort+"pruned_samples.csv")
        df_related = pd.read_csv(out_path+cohort+"related_samples.csv")
        df_input = pd.read_csv(out_path+cohort+"input_samples.csv")
        df_input.rename(columns={'#IID': 'IID'}, inplace=True)
        df_clinical=pd.read_csv(key_path+"/GP2_master_key_APRIL_2025.tsv", sep="\t")

        # now remove callrate and sex failed
        df_pruned_remaining = df_pruned[(df_pruned['step']!='callrate')&(df_pruned['step']!='sex')]
        df_pruned_remaining.reset_index(drop=True, inplace=True)
        print(f"df_pruned_remaining: {df_pruned_remaining.columns}")
        print(f"df_input: {df_input.columns}")
        df_pruned_samples = pd.merge(df_input, df_pruned_remaining, on='IID', how='inner')
        #first change name of IID column
        df_pruned_samples.rename(columns={'IID': 'GP2sampleID'}, inplace=True)
        df_pruned_samples_clinical = pd.merge(df_clinical, df_pruned_samples, on='GP2sampleID', how='inner')
        df_pruned_final = df_pruned_samples_clinical[['GP2sampleID', 'biological_sex_for_qc', 'GP2_phenotype_for_qc','step']]
        df_pruned_final.rename(columns={'step': 'pruned_reason'}, inplace=True)
        df_pruned_final['pruned_reason'] = 'duplicate'

        #first change name of IID column
        df_related.rename(columns={'IID1': 'GP2sampleID'}, inplace=True)
        df_related_all = pd.merge(df_pruned_final, df_related, on='GP2sampleID', how='inner')
        df_related_all = df_related_all[df_related_all['REL']=='duplicate']
        df_related_final = df_related_all[['GP2sampleID', 'IID2', 'biological_sex_for_qc', 'GP2_phenotype_for_qc','pruned_reason']]
        df_related_final.rename(columns={'biological_sex_for_qc': 'GP2sampleID_biological_sex_for_qc','GP2_phenotype_for_qc':'GP2sampleID_GP2_phenotype_for_qc'}, inplace=True)
        df_related_final = df_related_final.reset_index(drop=True)

        ###NOW duplicate
        temp_df1 = df_related_final[['GP2sampleID','IID2']]
        other_sample = df_clinical[df_clinical['GP2sampleID'].isin(temp_df1['IID2'])][['GP2sampleID','biological_sex_for_qc', 'GP2_phenotype_for_qc']]
        other_sample = other_sample.reset_index(drop=True)
        other_sample.rename(columns={'GP2sampleID':'IID2','biological_sex_for_qc': 'IID2_biological_sex_for_qc','GP2_phenotype_for_qc':'IID2_GP2_phenotype_for_qc'}, inplace=True)     
        other_sample_merge = pd.merge(df_related_final,other_sample, on='IID2', how='inner')
        temp_df = other_sample_merge.copy()
        temp_df[['GP2sampleID','IID2','GP2sampleID_biological_sex_for_qc',
            'GP2sampleID_GP2_phenotype_for_qc',
            'IID2_biological_sex_for_qc', 'IID2_GP2_phenotype_for_qc', 'pruned_reason']]=temp_df[['IID2','GP2sampleID','IID2_biological_sex_for_qc', 'IID2_GP2_phenotype_for_qc','GP2sampleID_biological_sex_for_qc',
            'GP2sampleID_GP2_phenotype_for_qc','pruned_reason']]
        concated_df = pd.concat([other_sample_merge,temp_df]).reset_index(drop=True)

        #########


        #Also duplicate the GP2sampleID and IID2 columns
        # temp_df = df_related_final.copy()
        # temp_df[['GP2sampleID','IID2']]=temp_df[['IID2','GP2sampleID']]
        # df_concat = pd.concat([df_related_final,temp_df]).reset_index(drop=True)
        # df_concat.rename(columns={'biological_sex_for_qc': 'GP2sampleID_biological_sex_for_qc','GP2_phenotype_for_qc':'GP2sampleID_GP2_phenotype_for_qc'}, inplace=True)

        # #also get sex and phenotype from IID2
        # iid2_sample = df_clinical[df_clinical['GP2sampleID'].isin(df_concat['IID2'])][['GP2sampleID','biological_sex_for_qc', 'GP2_phenotype_for_qc']]
        # iid2_sample = iid2_sample.reset_index(drop=True)
        # iid2_sample.rename(columns={'GP2sampleID': 'IID2'}, inplace=True)
        # df_concat1 = pd.merge(df_concat, iid2_sample, on='IID2', how='inner')
        # df_concat1.rename(columns={'biological_sex_for_qc': 'IID2_biological_sex_for_qc','GP2_phenotype_for_qc':'IID2_GP2_phenotype_for_qc'}, inplace=True)
    
        #also get sex only samples
        bbdp_pruned_sex = df_pruned[df_pruned['step']=='sex']
        bbdp_pruned_sex.rename(columns={'IID': 'GP2sampleID'}, inplace=True)
        # also get sex and pheno and
        bbdp_pruned_sex1 = pd.merge(df_clinical, bbdp_pruned_sex, on='GP2sampleID', how='inner')
        bbdp_pruned_sex1 = bbdp_pruned_sex1[['GP2sampleID', 'biological_sex_for_qc', 'GP2_phenotype_for_qc','step']]
        bbdp_pruned_sex1.rename(columns={'step': 'pruned_reason','biological_sex_for_qc':'GP2sampleID_biological_sex_for_qc', 'GP2_phenotype_for_qc':'GP2sampleID_GP2_phenotype_for_qc'}, inplace=True)
        #also add IID2 etc columns as blank
        bbdp_pruned_sex1[['IID2','IID2_biological_sex_for_qc','IID2_GP2_phenotype_for_qc']] = [['na','na','na'] for i in range(bbdp_pruned_sex1.shape[0])]
        #now concate the two data frames
        df_save = pd.concat([concated_df,bbdp_pruned_sex1]).reset_index(drop=True)
        re_order = ['GP2sampleID', 'IID2', 'GP2sampleID_biological_sex_for_qc', 'GP2sampleID_GP2_phenotype_for_qc','IID2_biological_sex_for_qc', 'IID2_GP2_phenotype_for_qc','pruned_reason']
        df_save = df_save[re_order]
        sorted_df = df_save.sort_values(by='GP2sampleID')
        sorted_df = sorted_df.reset_index(drop=True)
        sorted_df.to_csv(out_path+cohort+'_report.csv')        

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def read_json_file(file_path):
    """
    Reads a JSON file and returns the data as a Python object.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or list: The data from the JSON file, as a Python dictionary or list.
                      Returns None if an error occurs during file reading or JSON parsing.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def process_json(cohort, out_path):
    data = read_json_file(out_path+cohort+".json")

    # Extract all keys
    all_keys, keys1 = extract_keys(data)

    # Print the result
    print(f"All Keys: {keys1}")
    for key in keys1:
        try:
            res=pd.DataFrame(data[key])
            res.to_csv(f'{out_path}{cohort}{key}.csv', index=False)
        except Exception as e:
            print(f"An unexpected error occurred: {e} for key: {key}")

def extract_keys(data, parent_key='', separator='.'):
    """
    Recursively extracts all keys from a JSON object (dict or list).
    
    Args:
        data (dict or list): The JSON object to parse.
        parent_key (str): The accumulated key path for nested objects.
        separator (str): The separator to use for nested keys (e.g., 'parent.child').
    
    Returns:
        list: A list of all keys in the JSON object.
    """
    keys = []
    keys1 = []

    if isinstance(data, dict):  # If the current data is a dictionary
        for key, value in data.items():
            full_key = f"{parent_key}{separator}{key}" if parent_key else key
            keys.append(full_key)  # Add the current key
            keys1.append(full_key)
            keys.extend(extract_keys(value, full_key, separator))  # Recurse into the value

    elif isinstance(data, list):  # If the current data is a list
        for index, item in enumerate(data):
            full_key = f"{parent_key}[{index}]"  # Represent list items as "parent[index]"
            keys.extend(extract_keys(item, full_key, separator))  # Recurse into each item

    return keys, keys1


def background_task(prefix, cohort, request):
    """
    Simulates a long-running background task.
    """
    try:    
        logger.info(f"Starting background task for cohort: {prefix}_{cohort}")
        print(f"Starting background task for cohort: {prefix}_{cohort}")

        # for cohort in cohorts:
        command = f"genotools --pfile {request.geno_path}/{prefix}_{cohort} --out {request.out_path}{cohort} --ancestry --amr_het --all_sample --all_variant --prune_duplicated --full_output --ref_panel {request.ref_panel} --ref_labels {request.ref_labels} --model {request.model} --skip_fails"    
        # command = f"genotools --pfile {request.geno_path}/{prefix}_{cohort} --out {request.out_path}{cohort} --amr_het --all_sample --all_variant --prune_duplicated --full_output --ref_panel {request.ref_panel} --ref_labels {request.ref_labels} --model {request.model} --skip_fails"    
        send_email(subject_submitted, body_submitted+f"\n\nJob ID: {os.getpid()} For cohort: {prefix}_{cohort} Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"+"\n\nSubmission Command: "+command, sender_email, pat, request.user_email)

        print(f"cohort: {cohort}\n command: {command}")
        result = shell_do(command, log=True, return_log=True)  
        logger.info(f"Background task completed for cohort: {prefix}_{cohort} Completed with Results: {result}")
        print(f"Background task completed for cohort: {prefix}_{cohort} Completed with Results: {result}")

        logger.info(f"Starting json processing for cohort: {prefix}_{cohort}")
        print(f"Starting json processing for cohort: {prefix}_{cohort}")
        process_json(cohort, request.out_path)
        logger.info(f"Done json processing for cohort: {prefix}_{cohort}")
        print(f"Done json processing for cohort: {prefix}_{cohort}")
        # generate report
        logger.info(f"Generating report for cohort: {prefix}_{cohort}")
        print(f"Generating report for cohort: {prefix}_{cohort}")
        generate_report(cohort, request.out_path, request.key_path)
        send_email(subject_completed, body_completed+f"\n\nJob ID: {os.getpid()}  For cohort: {prefix}_{cohort} Is Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"+"\n\nSubmission Command: "+command+"\n\nResults: "+result, sender_email, pat, request.user_email)
        logger.info(f"Done generating report for cohort: {prefix}_{cohort}")
        print(f"Done generating report for cohort: {prefix}_{cohort}")

    except Exception as e:
        logger.error(f"Error Submitting background task: {e}")  
        send_email(subject_completed, body_completed+f"\n\nJob ID: {os.getpid()} For cohort: {prefix}_{cohort} Failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"+"\n\nSubmission Command: "+command+"\n\Error: "+{e}, sender_email, pat, request.user_email)
 

def get_cohorts(geno_path: str): # -> Tuple[str, List[str]]:
    all_files = os.listdir(geno_path)
    file_names = [f.split(".")[0] for f in os.listdir(geno_path) if os.path.isfile(os.path.join(geno_path, f))]
    # file_names = [f.split(".")[0] for f in all_files if os.path.isfile(os.path.join(geno_path, f))]
    prefix = '_'.join(file_names[0].split("_")[:-1])
    cohorts = [f.split("_")[-1].split(".")[0] for f in all_files if os.path.isfile(os.path.join(geno_path, f))]
    #make sure to remove duplicates and empty strings
    cohorts = set(cohorts)
    cohorts = [c for c in cohorts if c!='']
    return prefix, cohorts

    # return CohortsExtrator.extract_cohorts(geno_path)

@app.get("/")
async def health_check():
    return {"status": "healthy"}

class CarrierRequest(BaseModel):
    user_email: str  # Email of the user submitting the request
    geno_path: str  # Path to PLINK2 files prefix (without .pgen/.pvar/.psam extension)
    key_path: str  # Path to PLINK2 files prefix (without .pgen/.pvar/.psam extension)
    out_path: str  # Full output path prefix for the generated files
    ref_panel: str  # Path to the reference panel
    ref_labels: str  # Path to the reference labels
    model: str  # Path to the model file



@app.post("/prechecks/")
async def process_carriers(
    request: CarrierRequest,
    background_tasks: BackgroundTasks
):
    """
    Process carrier information from a single genotype file stored locally.
    Returns paths to the generated file.
    """
    try:
        out_dir = os.path.dirname(request.out_path)
        print(f"out_dir: {out_dir}")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        prefix, cohorts = get_cohorts(request.geno_path)
        print(f"prefix: {prefix}\ncohorts:{cohorts}")
        # processor.make_pool()
        for cohort in cohorts:
            print(f"Submitting background task for cohort: {cohort}")
            background_tasks.add_task(background_task, prefix, cohort, request)
        # results = manager.extract_carriers(
        #     geno_path=request.geno_path,
        #     snplist_path=request.snplist_path,
        #     out_path=request.out_path
        # )

        return {
            "status": "success",
            "outputs": cohorts
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {str(e)}\n\nTraceback: {error_trace}"
        )