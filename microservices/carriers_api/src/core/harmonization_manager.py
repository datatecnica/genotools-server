import os
from typing import Dict, Any, List, Optional
import pandas as pd

from src.core.harmonizer import AlleleHarmonizer
from src.core.data_repository import DataRepository


class HarmonizationPrecomputeManager:
    """
    Build and maintain a single, append-only harmonization map per dataset type and release.
    Incremental: only compute rows missing for each scope (ancestry/chromosome/prefix).
    """

    def __init__(self, base_output_dir: str):
        self.base_output_dir = os.path.expanduser(base_output_dir)
        self.repo = DataRepository()
        self.harmonizer = AlleleHarmonizer()

    def _map_path(self, release: str, data_type: str) -> str:
        rel_dir = os.path.join(self.base_output_dir, f"release{release}")
        os.makedirs(rel_dir, exist_ok=True)
        return os.path.join(rel_dir, f"{data_type}_harmonization_map.parquet")

    def _load_existing_map(self, map_path: str) -> pd.DataFrame:
        if os.path.exists(map_path):
            return self.repo.read_parquet(map_path)
        return pd.DataFrame()

    def _append_atomic(self, df: pd.DataFrame, map_path: str) -> None:
        tmp_path = f"{map_path}.tmp"
        self.repo.write_parquet(df, tmp_path, index=False)
        if os.path.exists(map_path):
            os.replace(tmp_path, map_path)
        else:
            os.rename(tmp_path, map_path)

    def _process_scope(self,
                       data_type: str,
                       release: str,
                       snplist_path: str,
                       geno_prefix: str,
                       ancestry: Optional[str] = None,
                       chromosome: Optional[str] = None,
                       existing: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        already = set()
        if existing is not None and not existing.empty:
            scope_mask = (existing['data_type'] == data_type) & (existing['release'] == release)
            if ancestry is not None:
                scope_mask &= (existing['ancestry'] == ancestry)
            else:
                scope_mask &= existing['ancestry'].isna()
            if chromosome is not None:
                scope_mask &= (existing['chromosome'] == chromosome)
            else:
                scope_mask &= existing['chromosome'].isna()
            if geno_prefix is not None:
                scope_mask &= (existing['geno_prefix'] == geno_prefix)
            already = set(existing.loc[scope_mask, 'variant_id_ref'].astype(str).tolist())

        snp_df = self.repo.read_csv(snplist_path, dtype={'chrom': str})
        snp_df['hg38'] = snp_df['hg38'].astype(str).str.strip().str.replace(' ', '')
        target_ids = set(snp_df['hg38'].str.upper().tolist())

        missing_ids = target_ids - already
        if not missing_ids:
            return pd.DataFrame()

        ref_missing = snp_df[snp_df['hg38'].str.upper().isin(missing_ids)].copy()
        # Write a temporary filtered reference file to limit scan to missing variants only
        tmp_ref_path = os.path.join(os.path.dirname(geno_prefix), ".harm_tmp_missing_ref.csv")
        ref_missing.to_csv(tmp_ref_path, index=False)

        common_prefix = os.path.join(os.path.dirname(geno_prefix), ".harm_tmp_common")
        common_ids_path = self.harmonizer._find_common_snps(geno_prefix, tmp_ref_path, common_prefix)  # noqa
        match_info_path = f"{common_prefix}_match_info.tsv"
        if not os.path.exists(match_info_path):
            return pd.DataFrame()

        match_df = pd.read_csv(match_info_path, sep='\t')
        if match_df.empty:
            return pd.DataFrame()

        rows = []
        for _, r in match_df.iterrows():
            a1_geno = str(r.get('a1_geno', '')).upper()
            a2_geno = str(r.get('a2_geno', '')).upper()
            a1_ref = str(r.get('a1_ref', '')).upper()
            a2_ref = str(r.get('a2_ref', '')).upper()
            match_type = r.get('match_type')
            needs_flip = match_type in ('flip', 'flip_swap')
            needs_swap = match_type in ('swap', 'flip_swap')
            try:
                chrom, pos, _, _ = r.get('variant_id_ref', ':::').split(':')
            except ValueError:
                chrom, pos = None, None
            rows.append({
                'data_type': data_type,
                'release': release,
                'ancestry': ancestry,
                'chromosome': chromosome,
                'geno_prefix': geno_prefix,
                'id_geno': r['id_geno'],
                'variant_id_ref': r['variant_id_ref'],
                'match_type': match_type,
                'chrom': chrom,
                'pos': pos,
                'a1_ref': a1_ref,
                'a2_ref': a2_ref,
                'a1_geno': a1_geno,
                'a2_geno': a2_geno,
                'needs_flip': needs_flip,
                'needs_swap': needs_swap,
                'snp_name_ref': r.get('snp_name_ref') if 'snp_name_ref' in r else None,
            })

        return pd.DataFrame(rows)

    def run(self,
            data_type: str,
            release: str,
            snplist_path: str,
            nba_dir: Optional[str] = None,
            wgs_prefix: Optional[str] = None,
            imputed_dir: Optional[str] = None,
            labels: Optional[List[str]] = None,
            chromosomes: Optional[List[str]] = None,
            dry_run: bool = False) -> Dict[str, Any]:
        map_path = self._map_path(release, data_type)
        existing = self._load_existing_map(map_path)

        if labels is None:
            labels = ['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN', 'MDE', 'SAS']
        if chromosomes is None:
            chromosomes = [str(i) for i in range(1, 23)] + ['X']

        to_append = []
        scopes_processed = 0

        if data_type == 'wgs':
            if not wgs_prefix:
                raise ValueError('wgs_prefix is required for data_type=wgs')
            df = self._process_scope('wgs', release, snplist_path, wgs_prefix, None, None, existing)
            if not df.empty:
                to_append.append(df)
            scopes_processed += 1

        elif data_type == 'nba':
            if not nba_dir:
                raise ValueError('nba_dir is required for data_type=nba')
            for label in labels:
                prefix = os.path.join(nba_dir, label, f"{label}_release{release}_vwb")
                df = self._process_scope('nba', release, snplist_path, prefix, ancestry=label, chromosome=None, existing=existing)
                if not df.empty:
                    to_append.append(df)
                scopes_processed += 1

        elif data_type == 'imputed':
            if not imputed_dir:
                raise ValueError('imputed_dir is required for data_type=imputed')
            for label in labels:
                for chrom in chromosomes:
                    prefix = os.path.join(imputed_dir, label, f"chr{chrom}_{label}_release{release}_vwb")
                    if not os.path.exists(f"{prefix}.pgen"):
                        continue
                    df = self._process_scope('imputed', release, snplist_path, prefix, ancestry=label, chromosome=chrom, existing=existing)
                    if not df.empty:
                        to_append.append(df)
                    scopes_processed += 1
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        appended_rows = 0
        if to_append:
            combined = pd.concat(to_append, ignore_index=True)
            appended_rows = len(combined)
            if not dry_run:
                if existing is not None and not existing.empty:
                    combined = pd.concat([existing, combined], ignore_index=True)
                    combined = combined.drop_duplicates(
                        subset=['data_type', 'release', 'ancestry', 'chromosome', 'geno_prefix', 'id_geno', 'variant_id_ref'],
                        keep='first'
                    )
                self._append_atomic(combined, map_path)

        return {
            'mapping_file': map_path,
            'scopes_processed': scopes_processed,
            'appended_rows': appended_rows,
            'existing_rows': 0 if existing is None or existing.empty else len(existing)
        }


