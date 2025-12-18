import argparse
import logging
import sys
from pathlib import Path

from src.processor import IDAT_BED_PED_Processor
from src.config import ProcessorConfig, ConfigurationError
from src.generate_jobs import generate_idat_ped_job_files, generate_ped_bed_job_files, generate_merge_beds_job_files

def setup_logging(log_file: str, verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    #create log-folder if it does not exist
    print(f"log_file: {log_file}")
    temp_path = str(log_file)
    directory_path = Path('/'.join(temp_path.split("/")[:-1]))
    print(f"directory_path: {directory_path}")
    directory_path.mkdir(parents=True, exist_ok=True)        
    #create log file if it does not exist
    log_file = Path(log_file)
    print(f"now log_file: {log_file}")
    log_file.touch(exist_ok=True) 

    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # return log_file
def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Process SNP metrics from IDAT files to parquet format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input path arguments (only required for normal processing, not variant reference)
    parser.add_argument(
        '--calc-flag',
        type=int,
        help='Steps to Run. 0 -> All steps, 1 -> IDAT -> PED, 2 -> PED -> BED, 3 -> BEDS Merge',
        default=0
    )

    parser.add_argument(
        '--log-file-path', 
        type=Path, 
        help='Specify complete log file name.')

    parser.add_argument(
        '--study-id',
        type=str,
        nargs='+',
        help='Study ID to process'
    )
    
    parser.add_argument(
        '--key-path',
        type=Path,
        help='Path to Key file'
    )
    
    parser.add_argument(
        '--fam-path',
        type=Path,
        help='Path to FAM files'
    )
    parser.add_argument(
        '--raw-plink-path',
        type=Path,
        help='Path to FAM files'
    )
    parser.add_argument(
        '--batch-folder-path',
        type=Path,
        help='Path to bash scripts'
    )
    parser.add_argument(
        '--exec-folder-path',
        type=str,
        help='Path to Plink Module'
    )
    parser.add_argument(
        '--idat-path',
        type=Path,
        help='Path to IDAT files'
    )

    parser.add_argument(
        '--barcodes-per-job',
        type=int,
        help='Number of barcodes to process per job',
        default=2
    )
    #ped-bed and beds-merge
    parser.add_argument(
        '--codes-per-job',
        type=int,
        help='Number of codes to process per job',
        default=2
    )
    #beds-merge
    parser.add_argument(
        '--clinical-key-dir',
        type=Path,
        help='Path to Clinical Key files'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        help='Number of threads to use for processing',
        default=1
    )
    #gke setup
    parser.add_argument(
        '--service-account-name',
        type=str,
        help='GCS PV Access Service Account Name',
        default="ksa-bucket-access"
    )
    parser.add_argument(
        '--k8s-namespace',
        type=str,
        help='K8s Namespace',
        default="kns-gtserver"
    )
    parser.add_argument(
        '--pv-claim',
        type=str,
        help='GCS PV claim name',
        default="gtserver-pvc"
    )

    parser.add_argument(
        '--gke-nodepools',
        type=str,
        help='K8S nodepools to use for the jobs',
        default="workflow-idat-ped-bed-nodepool"
    )
    # parser.add_argument(
    #     '--user-email',
    #     type=str,
    #     help='Valid Email of the user executing the workflow',
    #     default="syed@datatecnica.com"
    # )    

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    print(f"Arguments: {args}")
    try:
        # Create configuration for normal processing
        config = ProcessorConfig(
            calc_flag = args.calc_flag,
            log_file_path=args.log_file_path,
            study_id=args.study_id,
            key_path=args.key_path,
            fam_path=args.fam_path,
            raw_plink_path=args.raw_plink_path,
            batch_folder_path=args.batch_folder_path,
            num_threads=args.num_threads,
            idat_path=args.idat_path, #idat-ped
            barcodes_per_job=args.barcodes_per_job, #idat-ped
            exec_folder_path=args.exec_folder_path, #idat-ped
            codes_per_job=args.codes_per_job, #ped-bed and beds-merge
            clinical_key_dir=args.clinical_key_dir, #beds-merge
            service_account_name=args.service_account_name, #beds-merge
            k8s_namespace=args.k8s_namespace, #beds-merge
            pv_claim=args.pv_claim, #beds-merge
            gke_nodepools=args.gke_nodepools, #beds-merge
            # user_email = args.user_email
        )        
        
        
        # COMPLETE WORKFLOE
        
        #if args.calc_flag == 0 or args.calc_flag == 1: # 
        if args.calc_flag == 1: #             
            # Setup logging and update the config
            args.log_file_path = str(args.log_file_path)+"/idat_ped/pre_idat_ped.txt"
            config.log_file_path = Path(args.log_file_path)

            args.batch_folder_path = str(args.batch_folder_path) + "/idat_ped"
            config.batch_folder_path = Path(args.batch_folder_path)

            config.batch_folder_path.mkdir(parents=True, exist_ok=True)        
            setup_logging(args.log_file_path, args.verbose)
            logger = logging.getLogger(__name__)

            logger.info("Starting Preflight checks for IDAT -> PED Pipeline")

            processor = IDAT_BED_PED_Processor(config, num_threads=args.num_threads)
            
            # Process the barcode
            df_preprocess = processor.get_study_to_process()

            temp_path = str(args.log_file_path)
            log_directory_path = Path('/'.join(temp_path.split("/")[:-1]))
            key_name = args.key_path.name
            df_preprocess.to_csv(f"{log_directory_path}/pre_idat_ped_{key_name.split('.')[0]}.csv") #, index=False)
            
            if len(df_preprocess)>0:
                for index, row in df_preprocess.iterrows():
                    print(f"Study ID: {index}, Total Samples already processed: {row['samples_already_processed']}, Num of Barcodes to process: {len(row['barcode_list'])}, Barcodes to process: {row['barcode_list']}")
                    # barcode_list = row['barcode_list']
                    logger.info(f"Study ID: {index}, Total Samples already processed: {row['samples_already_processed']}, Num of Barcodes to process: {len(row['barcode_list'])}, Barcodes to process: {row['barcode_list']}")
                    logger.info(f"Now creating idat_ped script for study ID: {index}")
                    
                    job_files = processor.create_idat_ped_job_scripts(row['barcode_list'], index)
                    print(f"Generated job script files {job_files} for study ID: {index}")
                    logger.info(f"Done creating job script {job_files}")
                    print(f"Now generating YAML job files")
                    # generate_jobs.generate_idat_ped_job_files(args.batch_folder_path, index)
                    generate_idat_ped_job_files(args.batch_folder_path, args.exec_folder_path, index, args.k8s_namespace, args.pv_claim, args.service_account_name, args.gke_nodepools)#, args.user_email)
                    print(f"Done generating YAML job files")
                    logger.info(f"Done generating YAML job files")
            else:
                print(f"No Barcodes for given studies {args.study_id} to process")
                logger.info(f"No Barcodes for given studies {args.study_id} to process")
        else:
            print("Please enter valid calc_flag value: 1 -> IDAT -> PED step")           
        #elif args.calc_flag == 0 or args.calc_flag == 2:
        if args.calc_flag == 2:            
            # Setup logging and update the config
            args.log_file_path = str(args.log_file_path)+"/ped_bed/ped_bed.txt"
            config.log_file_path = Path(args.log_file_path)

            args.batch_folder_path = str(args.batch_folder_path) + "/ped_bed"
            config.batch_folder_path = Path(args.batch_folder_path)

            # config.batch_folder_path = str(args.batch_folder_path)+"/ped_bed"
            # config.batch_folder_path = Path(config.batch_folder_path)
            config.batch_folder_path.mkdir(parents=True, exist_ok=True)        

            setup_logging(args.log_file_path, args.verbose)
            logger = logging.getLogger(__name__)

            logger.info("Starting Preflight checks for BED -> BED Pipeline")
            processor = IDAT_BED_PED_Processor(config, num_threads=args.num_threads)

            #Get failed IDATs-> PEDS and generate job scripts
            missing_stats, updated_key_file = processor.get_failed_peds_generate_scritps()
            logger.info(f"Process Completed:\n\nMissing PED STATS:: {missing_stats}\n\nUpdated Key files path: {updated_key_file}")

            print(f"Now generating YAML job files for PED->BED conversion")
            # generate_jobs.generate_ped_bed_job_files(args.batch_folder_path)
            generate_ped_bed_job_files(args.batch_folder_path, args.exec_folder_path, args.k8s_namespace, args.pv_claim, args.service_account_name, args.gke_nodepools)#, args.user_email)
            print(f"Done generating YAML job files")
            logger.info(f"Done generating YAML job files")
        else:
            print("Please enter valid calc_flag value: 2 -> PED-BED step")           

                        
        # elif args.calc_flag == 0 or args.calc_flag == 3: 
        if args.calc_flag == 3: 
            # Setup logging and update the config
            args.log_file_path = str(args.log_file_path)+"/beds_merge/beds_merge.txt"
            config.log_file_path = Path(args.log_file_path)

            args.batch_folder_path = str(args.batch_folder_path) + "/beds_merge"
            config.batch_folder_path = Path(args.batch_folder_path)
            
            # config.batch_folder_path = str(args.batch_folder_path)+"/beds_merge"
            # config.batch_folder_path = Path(config.batch_folder_path)
            config.batch_folder_path.mkdir(parents=True, exist_ok=True)        

            setup_logging(args.log_file_path, args.verbose)
            logger = logging.getLogger(__name__)
            logger.info(f"Starting Preflight checks for BEDS-MERGE Pipeline")
            processor = IDAT_BED_PED_Processor(config, num_threads=args.num_threads)

            df_beds_merge_stats = processor.get_failed_ped_bed()

            for index, row in df_beds_merge_stats.iterrows():
                logger.info(f"For cohort {index}, There are {row['missing_cnt']} Missing BEDS: {row['missing_beds']} and beds list: {row['beds_to_merge']}")

                logger.info(f"Started creating merge BEDs scripts for cohort {index}")

                all_jobs_scripts_path = processor.create_merge_beds_script()
                logger.info(f"Done creating merge BED scripts: {all_jobs_scripts_path}")

                print(f"Now generating YAML job files for Merge BEDs")
                # generate_jobs.generate_merge_beds_job_files(args.batch_folder_path)
                generate_merge_beds_job_files(args.batch_folder_path, args.exec_folder_path, args.k8s_namespace, args.pv_claim, args.service_account_name, args.gke_nodepools)#, args.user_email)
                print(f"Done generating BEDS-MERGE YAML job files")
                logger.info(f"Done generating BEDS-MERGE YAML job files")
            
        else:
            print("Please enter valid calc_flag value: 3 -> BEDS-MERGE step")           
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 