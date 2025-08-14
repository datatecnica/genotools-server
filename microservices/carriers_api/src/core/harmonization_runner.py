import os
import pandas as pd
from typing import Dict, Any, List, Optional

from src.core.data_repository import DataRepository
from src.core.plink_operations import UpdateAllelesCommand, SwapAllelesCommand, ExportCommand
from src.core.manager import CarrierAnalysisManager


class HarmonizationRunner:
    """
    Use a precomputed harmonization map to materialize harmonized PLINK exports
    and produce finalized parquet outputs (var_info, carriers_int, carriers_string).
    Also merges across chromosomes and ancestries to produce dataset-level outputs.
    """

    def __init__(self, mapping_file: str):
        self.mapping_file = os.path.expanduser(mapping_file)
        self.repo = DataRepository()
        self.manager = CarrierAnalysisManager()

    def _filter_scope(self, df: pd.DataFrame, data_type: str, release: str,
                      ancestry: Optional[str], chromosome: Optional[str], geno_prefix: Optional[str]) -> pd.DataFrame:
        mask = (df['data_type'] == data_type) & (df['release'] == release)
        mask &= (df['ancestry'] == ancestry) if ancestry is not None else df['ancestry'].isna()
        mask &= (df['chromosome'] == chromosome) if chromosome is not None else df['chromosome'].isna()
        mask &= (df['geno_prefix'] == geno_prefix) if geno_prefix is not None else True
        return df[mask].copy()

    def _write_helper_files(self, scope_df: pd.DataFrame, tmp_dir: str) -> Dict[str, str]:
        os.makedirs(tmp_dir, exist_ok=True)
        extract_ids = os.path.join(tmp_dir, 'extract_ids.txt')
        update_alleles = os.path.join(tmp_dir, 'update_alleles.txt')
        swap_snps = os.path.join(tmp_dir, 'swap_snps.txt')

        # extract ids
        scope_df['id_geno'].drop_duplicates().to_csv(extract_ids, index=False, header=False)

        # update alleles for flips
        flip_df = scope_df[scope_df['needs_flip'] == True]  # noqa
        if not flip_df.empty:
            # id oldA1 oldA2 newA1 newA2
            upd = flip_df[['id_geno', 'a1_geno', 'a2_geno']].copy()
            comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
            upd['new_a1'] = upd['a1_geno'].map(lambda x: comp.get(str(x).upper(), str(x).upper()))
            upd['new_a2'] = upd['a2_geno'].map(lambda x: comp.get(str(x).upper(), str(x).upper()))
            upd.to_csv(update_alleles, sep='\t', index=False, header=False)
        else:
            update_alleles = None

        # swap alleles
        swap_df = scope_df[scope_df['needs_swap'] == True]  # noqa
        if not swap_df.empty:
            # ID, new A1 target = a1_ref
            sw = swap_df[['id_geno', 'a1_ref']].copy()
            sw.to_csv(swap_snps, sep='\t', index=False, header=False)
        else:
            swap_snps = None

        return {
            'extract_ids': extract_ids,
            'update_alleles': update_alleles,
            'swap_snps': swap_snps,
        }

    def _build_subset_snp_csv(self, scope_df: pd.DataFrame, snplist_path: str, out_csv: str) -> None:
        # split variant_id_ref into chrom,pos,a1,a2
        parts = scope_df['variant_id_ref'].str.split(':')
        subset = pd.DataFrame({
            'id': scope_df['id_geno'],
            'chrom': parts.str[0],
            'pos': parts.str[1],
            'a1': parts.str[2],
            'a2': parts.str[3],
        })
        # add snp_name if present in snplist
        try:
            ref = self.repo.read_csv(os.path.expanduser(snplist_path))
            if 'hg38' in ref.columns and 'snp_name' in ref.columns:
                ref['hg38'] = ref['hg38'].astype(str).str.upper()
                scope_df_work = scope_df.copy()
                scope_df_work['variant_id_ref'] = scope_df_work['variant_id_ref'].astype(str).str.upper()
                # map variant_id_ref to snp_name, then merge into subset by id_geno linkage
                name_map = scope_df_work.merge(ref[['hg38', 'snp_name']], left_on='variant_id_ref', right_on='hg38', how='left')
                name_map = name_map[['id_geno', 'snp_name']].drop_duplicates()
                subset = subset.merge(name_map, left_on='id', right_on='id_geno', how='left')
                subset.drop(columns=['id_geno'], inplace=True, errors='ignore')
        except Exception:
            pass
        subset.to_csv(out_csv, index=False)

    def _export_scope(self, geno_prefix: str, helpers: Dict[str, str], plink_out: str) -> None:
        current = geno_prefix
        if helpers.get('update_alleles'):
            updated = f"{plink_out}_upd"
            UpdateAllelesCommand(current, helpers['update_alleles'], updated).execute()
            current = updated
        if helpers.get('swap_snps'):
            swapped = f"{plink_out}_swp"
            SwapAllelesCommand(current, helpers['swap_snps'], swapped).execute()
            current = swapped
        args = [f"--extract {helpers['extract_ids']}"] if helpers.get('extract_ids') else []
        ExportCommand(current, plink_out, additional_args=args).execute()

    def harmonize_dataset(self,
                          data_type: str,
                          release: str,
                          snplist_path: str,
                          nba_dir: Optional[str] = None,
                          wgs_prefix: Optional[str] = None,
                          imputed_dir: Optional[str] = None,
                          labels: Optional[List[str]] = None,
                          chromosomes: Optional[List[str]] = None,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
        output_dir = os.path.expanduser(output_dir) if output_dir else os.path.expanduser(f"~/gcs_mounts/genotools_server/harmonized/release{release}/{data_type}")
        os.makedirs(output_dir, exist_ok=True)
        mapping = self.repo.read_parquet(self.mapping_file)

        if labels is None:
            labels = ['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN', 'MDE', 'SAS']
        if chromosomes is None:
            chromosomes = [str(i) for i in range(1, 23)] + ['X']

        results_by_label: Dict[str, Dict[str, str]] = {}

        if data_type == 'wgs':
            scope_df = self._filter_scope(mapping, 'wgs', release, ancestry=None, chromosome=None, geno_prefix=wgs_prefix)
            if scope_df.empty:
                raise ValueError('No scope rows found in mapping for WGS')
            tmp_dir = os.path.join(output_dir, 'tmp_wgs')
            helpers = self._write_helper_files(scope_df, tmp_dir)
            plink_out = os.path.join(output_dir, f"wgs_release{release}_snps")
            self._export_scope(wgs_prefix, helpers, plink_out)
            subset_csv = os.path.join(output_dir, f"wgs_release{release}_subset_snps.csv")
            self._build_subset_snp_csv(scope_df, snplist_path, subset_csv)
            out_prefix = os.path.join(output_dir, f"release{release}")
            files = self.manager.carrier_extractor._process_traw_data(subset_csv, plink_out, out_prefix)
            results_by_label['WGS'] = files

        elif data_type == 'nba':
            for label in labels:
                geno_prefix = os.path.join(nba_dir, label, f"{label}_release{release}_vwb")
                scope_df = self._filter_scope(mapping, 'nba', release, ancestry=label, chromosome=None, geno_prefix=geno_prefix)
                if scope_df.empty:
                    continue
                tmp_dir = os.path.join(output_dir, f"tmp_{label}")
                helpers = self._write_helper_files(scope_df, tmp_dir)
                plink_out = os.path.join(output_dir, f"{label}_release{release}_snps")
                self._export_scope(geno_prefix, helpers, plink_out)
                subset_csv = os.path.join(output_dir, f"{label}_release{release}_subset_snps.csv")
                self._build_subset_snp_csv(scope_df, snplist_path, subset_csv)
                out_prefix = os.path.join(output_dir, label, f"{label}_release{release}")
                os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
                files = self.manager.carrier_extractor._process_traw_data(subset_csv, plink_out, out_prefix)
                results_by_label[label] = files

        elif data_type == 'imputed':
            for label in labels:
                per_chrom_results: Dict[str, Dict[str, str]] = {}
                for chrom in chromosomes:
                    geno_prefix = os.path.join(imputed_dir, label, f"chr{chrom}_{label}_release{release}_vwb")
                    if not os.path.exists(f"{geno_prefix}.pgen"):
                        continue
                    scope_df = self._filter_scope(mapping, 'imputed', release, ancestry=label, chromosome=chrom, geno_prefix=geno_prefix)
                    if scope_df.empty:
                        continue
                    tmp_dir = os.path.join(output_dir, f"tmp_{label}_chr{chrom}")
                    helpers = self._write_helper_files(scope_df, tmp_dir)
                    plink_out = os.path.join(output_dir, f"{label}_chr{chrom}_release{release}_snps")
                    self._export_scope(geno_prefix, helpers, plink_out)
                    subset_csv = os.path.join(output_dir, f"{label}_chr{chrom}_release{release}_subset_snps.csv")
                    self._build_subset_snp_csv(scope_df, snplist_path, subset_csv)
                    out_prefix = os.path.join(output_dir, label, f"{label}_release{release}_chr{chrom}")
                    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
                    files = self.manager.carrier_extractor._process_traw_data(subset_csv, plink_out, out_prefix)
                    per_chrom_results[chrom] = files
                if per_chrom_results:
                    combined_prefix = os.path.join(output_dir, label, f"{label}_release{release}")
                    files = self.manager.carrier_extractor._combine_chromosomes(per_chrom_results, combined_prefix)
                    results_by_label[label] = files
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        # If multiple labels present, combine across labels
        dataset_outputs: Dict[str, str] = {}
        if data_type in ('nba', 'imputed') and results_by_label:
            combined_out = os.path.join(output_dir, f"{data_type}_release{release}_combined")
            dataset_outputs = self.manager.carrier_combiner.combine_carrier_files(
                results_by_label, key_file=None, out_path=combined_out, track_probe_usage=True
            )
        elif data_type == 'wgs' and 'WGS' in results_by_label:
            dataset_outputs = results_by_label['WGS']

        return {
            'per_label_outputs': results_by_label,
            'dataset_outputs': dataset_outputs,
            'output_dir': output_dir,
        }


