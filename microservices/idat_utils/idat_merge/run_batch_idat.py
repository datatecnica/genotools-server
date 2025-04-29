import os
import sys
import shutil
import pandas as pd
import batch_idat

def main():

    # hard coded paths
    # assuming mounting to dirs?
    iaap = f'./gp2_genotools_data/ilmn_files/iaap-cli/iaap-cli'
    bpm = f'./gp2_genotools_data/ilmn_files/NeuroBooster_20042459_A2.bpm'
    egt = f'./gp2_genotools_data/ilmn_files/recluster_09272022.egt'
    map_file = f'./gp2_genotools_data/ped_bed/NeuroBooster_20042459_A2.map'

    output_dir = "./gp2_genotools_data/output"

     # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # input args
    key_path = sys.argv[1]
    study = sys.argv[2]

    key = pd.read_csv(key_path, sep='\t', low_memory=False)
    key = key[key['study']==study]

    # check if previous samples exist
    samples = pd.read_csv(f'./gp2_genotools_data/merged_by_cohort_r10/GP2_merge_{study}.fam', sep='\s+', header=None)
    list_included = list(samples[1])
    # the samples we still need to process are the ones that were NOT included in the prevous merged file
    key = key[~key['GP2sampleID'].isin(list_included)]

    raw_plink_path = f'./gp2_genotools_data/ped_bed'
    missing_idat_dir = f'./gp2_genotools_data/missing'
    idat_path = f'./gp2_idats/${{BARCODE}}'

    # convert idat to ped files
    batch_idat.convert_idat_to_ped(key, study, iaap, bpm, egt, raw_plink_path, idat_path, missing_idat_dir, map_file)

    # convert ped to bed files
    batch_idat.convert_ped_to_bed(key, study, raw_plink_path, missing_idat_dir)

    # write files to update id (from clinical data)
    clin_key_dir = './gp2_genotools_data/clinical'
    key[['FID','IID', 'FID', 'GP2sampleID']].to_csv(f'{clin_key_dir}/update_ids_{study}.txt', sep='\t', header=False, index=False)

    # merge cohorts together
    batch_idat.merge_bed_files(study, raw_plink_path, clin_key_dir)

    # merge old data with newly processed data?
    # study.bed vs study_extra.bed?


if __name__ == "__main__":
    exit(main())