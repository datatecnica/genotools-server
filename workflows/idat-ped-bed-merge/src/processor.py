import pandas as pd
import os
import shutil
import logging

from itertools import zip_longest
from pathlib import Path

from .config import ProcessorConfig

# Helper function to chunk the list into groups of n
def chunk_list(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    args = [iter(iterable)] * n
    return zip_longest(*args)


class IDAT_BED_PED_Processor:
    """Processes genetic data from IDAT to parquet via DRAGEN pipeline.
    
    This class encapsulates the complete workflow:
    1. IDAT -> PED conversion 
    2. PED -> BED conversion 
    3. Merge BEDS
    """
    def __init__(self, config: ProcessorConfig, num_threads: int = 1):
        self.config = config
        self.num_threads = num_threads
        self.logger = self._setup_logging()

    def get_study_to_process(self) -> list:
        """
        Returns list of barcodes to process for the given study after removing all samples already processed for this study
        Input: key_path, fam_path and study_id
        Output: key_file with list of samples to process
        """
        #read the key file and retain only samples for the given study_id
        key1 = pd.read_csv(self.config.key_path, sep = '\t', low_memory = False)
        all_stats = {}
        for study in self.config.study_id:
            key = key1[key1['study'] == study ] #self.config.study_id]
            all_samples = len(key)
            if os.path.exists(f'{self.config.fam_path}/GP2_merge_{study}.fam'):
                test = pd.read_csv(f'{self.config.fam_path}/GP2_merge_{study}.fam', sep = '\s+', header = None)
                list_included = list(test[1])
                samples_included = len(list_included)
                # print(len(list_included))  
            
                key = key[~key['GP2sampleID'].isin(list_included)]             
            else:
                list_included = []
                samples_included = 0
            # Create a list of all the barcodes we need to call
            # key = key[key['study']==self.config.study_id]
            key = key[key['study']==study]
            # print(len(key))
            barcode_list = list(set(list(key['SentrixBarcode_A'])))
            all_stats[study] = {"all_samples": all_samples, "samples_already_processed": samples_included, "samples_to_process": key.shape[0], "barcode_list": barcode_list}
            self.logger.info(f"study_id: {study}, all_samples: {all_samples}, samples_already_processed: {samples_included}, samples_to_process: {key.shape[0]}, barcode_list: {barcode_list}")
        if len(all_stats) == 0:
            df_preprocess = pd.DataFrame()
        else:
            df_preprocess = pd.DataFrame.from_dict(all_stats, orient='index')
        return df_preprocess #all_samples, barcode_list
    def create_idat_ped_job_scripts(self, barcode_list: list, study: str) -> str:
        job_files = []
        count = 0
        for chunk in chunk_list(barcode_list, self.config.barcodes_per_job):
            # Filter out any None values from the last incomplete chunk
            codes = [code for code in chunk if code is not None]
            
            # Generate a unique job name
            count += 1
            # job_name = f'idattoped{self.config.study_id.lower()}{count}'
            job_name = f'idattoped-{study.lower()}-{count}'
            
            script=""

            # Add commands for each code in the chunk
            for code in codes:
                script += f"""
                # Make analysis script executable and run for a specific code
                #!/bin/bash
                ./tmp/exec/iaap-cli/iaap-cli gencall /tmp/exec/NeuroBooster_20042459_A2.bpm /tmp/exec/recluster_09272022.egt  {self.config.raw_plink_path}/ -f {self.config.idat_path}/{code} -p -t {self.config.num_threads}                
                """
            #save the script to a file
            script_path = f'{self.config.batch_folder_path}/{job_name}.sh'
            job_files.append(script_path)
            with open(script_path, 'w') as f:
                f.write(script)
                f.close()
            self.logger.info(f"Job script {script_path} generated for IDAT -> PED conversion for {study}.")
        #also save the list of job files
        all_jobs_scripts_path = f'{self.config.batch_folder_path}/all_idat_ped_{study.lower()}_scripts.txt'
        with open(all_jobs_scripts_path, 'w') as f:
            f.write("\n".join(job_files))
            f.close()
        self.logger.info(f"Done creating all {len(job_files)} Job script generated for IDAT -> PED conversion for {study}.")
        return job_files
    def get_study_to_process_cohort(self, cohort: str) -> list:
        """
        Returns list of barcodes to process for the given study after removing all samples already processed for this study
        Input: key_path, fam_path and study_id
        Output: key_file with list of samples to process
        """
        #read the key file and retain only samples for the given study_id
        key = pd.read_csv(self.config.key_path, sep = '\t', low_memory = False)
        key = key[key['study'] == cohort]
        all_samples = len(key)

        # read the fam file for the given study_id and retain only samples in the key file which are not in the fam file
        if os.path.exists(f'{self.config.fam_path}/GP2_merge_{cohort}.fam'):
            test = pd.read_csv(f'{self.config.fam_path}/GP2_merge_{cohort}.fam', sep = '\s+', header = None)
            list_included = list(test[1])
            samples_included = len(list_included)
            # print(len(list_included))  

            key = key[~key['GP2sampleID'].isin(list_included)]             
        else:
            samples_included = 0
            # Create a list of all the barcodes we need to call
            key = key[key['study']==cohort]
        # print(len(key))
        barcode_list = list(set(list(key['SentrixBarcode_A'])))

        return all_samples, samples_included, barcode_list, key

    def get_failed_peds_generate_scritps(self) -> list:
        """
        Returns list of barcodes to process for the given study after removing all samples already processed for this study
        Input: key_path, fam_path and study_id
        Output: key_file with list of samples to process
        """
        
        temp_path = str(self.config.raw_plink_path)
        missing_peds_dir = "/".join(temp_path.split('/')[:-1])+'/missing_peds'
        if not os.path.exists(missing_peds_dir):
            os.makedirs(missing_peds_dir)
        # key = pd.read_csv(self.config.key_path, sep = '\t', low_memory = False)
        # also create file to list all scripts
        all_jobs_scripts_path = f'{self.config.batch_folder_path}/all_ped_bed_scripts.txt'
        with open(all_jobs_scripts_path, 'w') as f:
            pass  # No content is written, resulting in an empty file
                
        # missing_peds = []
        missing_stats = {}
        # for cohort in [self.config.study_id]:
        self.logger.info(f'Processing Cohorts: {self.config.study_id}')
        for cohort in self.config.study_id:
        #First check how many samples and barcodes we are expectiong for this step
            all_samples, samples_included, barcode_list, key = self.get_study_to_process_cohort(cohort)
            self.logger.info(f"Number of samples expected for {cohort}: {all_samples}, Samples arlready processed: {samples_included}, Barcodes to process: {barcode_list}, key: {key.shape[0]}")
            df = key[key['study']==cohort]
            
            missing_cnt = 0
            missing_peds = []

            for filename in df.IID:
                # ped = f'{raw_plink_path}/{filename}.ped'
                ped = f'{self.config.raw_plink_path}/{filename}.ped'
                # out_map = f'{raw_plink_path}/{filename}.map'
                out_map = f'{self.config.raw_plink_path}/{filename}.map'
                if os.path.isfile(ped):
                    shutil.copyfile(src=f'{self.config.raw_plink_path}/NeuroBooster_20042459_A2.map', dst=out_map)
                else:
                    missing_cnt += 1
                    missing_peds.append(filename)
                    
            self.logger.info(f"There is {missing_cnt} missing ped file for {cohort}.")
            missing_stats[cohort] = missing_cnt


            if len(missing_peds)>0:
                #Get updated key file            
                key = key[~key['IID'].isin(missing_peds)]
                with open(f'{missing_peds_dir}/missing_peds_{cohort}.txt', 'w') as f:
                    for m_ped in missing_peds:
                        f.write(f'{m_ped}\n')
                f.close()
            if missing_stats:
                self.logger.info(f"For Cohort: {cohort}, we have {all_samples} samples and {barcode_list} barcodes")
                self.logger.info(f"For Cohort: {cohort}, we have missing peds: {missing_peds} and their Stats for missing/failed IDATs: {missing_stats}")
            key.to_csv(f'{self.config.batch_folder_path}/{cohort}_key_file_idat_ped.txt', sep = '\t', index = False)
            # print(f"Key file updated for {cohort}.")
            self.logger.info(f"Updated key file for cohort: {cohort} is saved at: {self.config.batch_folder_path}/{cohort}_key_file_idat_ped.txt.")
            self.logger.info(f"For cohort: {cohort}, Now creating scripts for PED -> BED for created ped files.")
            if len(key)>1:
                job_files = self.create_ped_bed_job_scriptV1(list(key['IID']), cohort)
                with open(all_jobs_scripts_path, 'a') as f:
                    self.logger.info(f"Appending job files {job_files} to {all_jobs_scripts_path}.")
                    for job in job_files:
                        f.write(job+"\n")
                    f.close()
                self.logger.info(f"Done creating all scripts {job_files} for PED -> BED for cohort: {cohort}.")
        # return missing_peds, missing_stats, f'{self.config.batch_folder_path}'    
        return missing_stats, f'{self.config.batch_folder_path}'    
    def create_ped_bed_job_scriptV1(self, filename_list: list, study: str) -> str:
        job_files = []
        count = 0
        for chunk in chunk_list(filename_list, self.config.codes_per_job):
            # Filter out any None values from the last incomplete chunk
            codes = [code for code in chunk if code is not None]
            
            # Generate a unique job name
            count += 1
            job_name = f'pedtobed-{study.lower()}-{count}'
                
            script = f"""
                #!/bin/bash
            """
            # Add commands for each code in the chunk
            for code in codes:
                script += f"""
                # Make analysis script executable and run for a specific code
                ./home/plink_package/bin/plink1.9/plink --file {self.config.raw_plink_path}/{code} --make-bed --out {self.config.raw_plink_path}/{code}                
                """
            #save the script to a file
            script_path = f'{self.config.batch_folder_path}/{job_name}.sh'
            job_files.append(script_path)
            with open(script_path, 'w') as f:
                f.write(script)
                f.close()
            self.logger.info(f"Job script {script_path} generated for PED -> BED conversion for {study}.")
        #also save the list of job files
        self.logger.info(f"Done creating all {len(job_files)} Job script generated for PED -> BED conversion for cohort: {study}.")
        return job_files
    def get_failed_ped_bed(self) -> list:
        """
        Returns list of barcodes to process for the given study after removing all samples already processed for this study
        Input: key_path, fam_path and study_id
        Output: key_file with list of samples to process
        """
        
        temp_path = str(self.config.raw_plink_path)
        missing_idat_dir = "/".join(temp_path.split('/')[:-1])+'/missing_beds'
        if not os.path.exists(missing_idat_dir):
            os.makedirs(missing_idat_dir)
        # key = pd.read_csv(self.config.key_path, sep = '\t', low_memory = False)
        
        # missing_beds = []
        missing_stats = {}
        all_stats = {}
        for cohort in self.config.study_id:
            missing_beds = []
            #First check how many samples and barcodes we are expectiong for this step
            
            # all_samples, barcode_list, key = self.get_study_to_process_cohort(cohort)
            all_samples, samples_included, barcode_list, key  = self.get_study_to_process_cohort(cohort)
            df = key[key['study']==cohort]
            
            missing_cnt = 0
            with open(f"{self.config.raw_plink_path}/merge_bed_{cohort}.list", 'w') as f:
                for filename in df.IID:
                    # ped = f'{raw_plink_path}/{filename}.ped'
                    bed = f'{self.config.raw_plink_path}/{filename}.bed'
                    # out_map = f'{raw_plink_path}/{filename}.map'
                    # out_map = f'{self.config.raw_plink_path}/{filename}.map'
                    if os.path.isfile(bed):
                        f.write(f'{bed}\n')
                    else:
                        missing_cnt += 1
                        missing_beds.append(filename)
            f.close()            
            self.logger.info(f"There is {missing_cnt} missing bed file for {cohort}.")
            missing_stats[cohort] = missing_cnt


            with open(f'{missing_idat_dir}/missing_beds_{cohort}.txt', 'w') as f:
                for m_bed in missing_beds:
                    f.write(f'{m_bed}\n')
            f.close()
            if missing_stats:
                self.logger.info(f"For Cohort {cohort}, {all_samples} samples and {barcode_list} barcodes")
                self.logger.info(f"Missing beds: {missing_beds} and their Stats for missing/failed IDATs: {missing_stats}")
            #Get original key file
            # key = pd.read_csv(self.config.key_path, sep = '\t', low_memory = False)
            self.logger.info(f"Loaded original key file: {self.config.key_path}.")
            self.logger.info(f"Now Loading clinical key for each study.")
            # for cohort in [self.config.study_id]:
            # df = key[key['study'] == cohort]
            df[['FID','IID', 'FID', 'GP2sampleID']].to_csv(f'{self.config.clinical_key_dir}/update_ids_{cohort}.txt', sep='\t', header=False, index=False)
            df[['FID', 'GP2sampleID', 'pheno']].to_csv(f'{self.config.clinical_key_dir}/update_pheno_{cohort}.txt', sep='\t', header=False, index=False)
            df[['FID', 'GP2sampleID', 'sex_for_qc']].to_csv(f'{self.config.clinical_key_dir}/update_sex_{cohort}.txt', sep='\t', header=False, index=False)
            self.logger.info(f"Updated clinical key files for {cohort} in {self.config.clinical_key_dir}.")
            # Also creates a list of all the files that we need to merge together
            missing_beds_1 = []
            # for cohort in [self.config.study_id]:
            # df = key[key['study'] == cohort]
            merged_bed_list = []
            with open(f"{self.config.raw_plink_path}/merge_bed_{cohort}.list", 'w') as f:
                for filename in df.IID:
                    bed = f'{self.config.raw_plink_path}/{filename}'
                    merged_bed_list.append(f'{bed}.bed')
                    if os.path.isfile(f'{bed}.bed'):
                        f.write(f'{bed}\n')
                    else:
                        print(f'{bed} does not exist in current directory!!!')
                        missing_beds_1.append(filename)
            f.close()

            all_stats[cohort] = {"missing_cnt": missing_cnt, "missing_beds": missing_beds, "beds_to_merge": merged_bed_list, "missing_beds_1": missing_beds_1, 'merged_bed_list': f"{self.config.raw_plink_path}/merge_bed_{cohort}.list"}
        df_beds_merge_stats = pd.DataFrame.from_dict(all_stats, orient='index')
        return df_beds_merge_stats #missing_beds, missing_stats, missing_beds_1, f"{self.config.raw_plink_path}/merge_bed_{cohort}.list"    

    def create_merge_beds_script(self) -> str:
        print("Creating job script for Merging all bed files now.")
        count = 0
        job_files = []

        # Set up the barcodes to process above

        # Loop through chromosomes and create separate jobs
        for study in self.config.study_id:
            # Generate a unique job name
            if os.path.getsize(f"{self.config.raw_plink_path}/merge_bed_{study}.list") > 0:
                count += 1
                job_name = f'mergebycohort-{study.lower()}-{count}'
                #also add key name for reference
                key_ref = str(self.config.key_path)
                key_subfolder = key_ref.split("/")[-1].split(".")[0]
                out_folder = str(self.config.raw_plink_path)
                out_folder = '/'.join(out_folder.split("/")[:-1]) 
                # out_path = f'{self.config.raw_plink_path}/merged_by_cohort/{key_subfolder}/GP2_merge_{study}'
                out_folder = f'{out_folder}/merged_by_cohort/{key_subfolder}'
                out_path1 = Path(out_folder)
                out_path1.mkdir(parents=True, exist_ok=True)

                # out_path = f'{out_folder}/merged_by_cohort/{key_subfolder}/GP2_merge_{study}'
                out_path = f'{out_folder}/GP2_merge_{study}'
                # out_folder = f'{out_folder}/merged_by_cohort/{key_subfolder}'
                # out_path1=Path(out_path)

                # out_path1.mkdir(parents=True, exist_ok=True)
                # Create a script for the specific chromosome
                script = f"""
                    #!/bin/bash
                    /home/plink_package/bin/plink1.9/plink --merge-list {self.config.raw_plink_path}/merge_bed_{study}.list --update-ids {self.config.clinical_key_dir}/update_ids_{study}.txt --make-bed --out {out_path}
                """
                #save the script to a file
                script_path = f'{self.config.batch_folder_path}/{job_name}.sh'
                job_files.append(script_path)
                with open(script_path, 'w') as f:
                    f.write(script)
                    f.close()
                self.logger.info(f"Job script {script_path} generated for Merging all bed files for {study}.")
            else:
                self.logger.info(f"No bed files to merge for {study}, skipping job script creation.")
        #also save the list of job files
        all_jobs_scripts_path = f'{self.config.batch_folder_path}/all_bed_merge_scripts.txt'
        with open(all_jobs_scripts_path, 'w') as f:
            f.write("\n".join(job_files))
            f.close()
        self.logger.info(f"Done creating all {len(job_files)} Job script for Merging bed files for all cohorts.")

        return all_jobs_scripts_path 

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the processor."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
