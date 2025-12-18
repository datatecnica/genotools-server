import streamlit as st
import pandas as pd
from typing import List, Dict, Optional


class CarriersConfig:
    """config settings for the application."""
    NON_VARIANT_COLUMNS: List[str] = ['IID', 'ancestry', 'study']
    WILD_TYPE: str = 'WT/WT'
    MISSING_GENOTYPE: str = './.'

class GenotypeMatcher:
    """handler for genotype matching and carrier status"""

    @staticmethod
    def is_carrier(genotype: str) -> bool:
        """check if a genotype indicates carrier status"""
        return genotype not in [CarriersConfig.WILD_TYPE, CarriersConfig.MISSING_GENOTYPE]

    @staticmethod
    def is_homozygous(genotype: str) -> bool:
        """check if a genotype is homozygous"""
        if genotype == CarriersConfig.MISSING_GENOTYPE:
            return False
        alleles = genotype.split('/')
        return alleles[0] == alleles[1] and alleles[0] != "."


class CarrierDataProcessor:
    """processes carrier data and manages carrier status determination"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.variants = self._get_variants()

    def _get_variants(self) -> List[str]:
        """xtract variant columns from the dataframe"""
        return [col for col in self.df.columns if col not in CarriersConfig.NON_VARIANT_COLUMNS]

    def get_carrier_status(self, row: pd.Series, selected_variants: List[str],
                           zygosity_filter: str = 'All') -> Dict[str, str]:
        """determine carrier status for a given row based on selected variants and zygosity filter"""
        status = {}
        has_carrier = False

        for variant in selected_variants:
            if variant in row:
                genotype = row[variant]
                if GenotypeMatcher.is_carrier(genotype):
                    if self._matches_zygosity_filter(genotype, zygosity_filter):
                        has_carrier = True

        if has_carrier:
            status = {variant: row[variant]
                      for variant in selected_variants if variant in row}

        return status

    def _matches_zygosity_filter(self, genotype: str, zygosity_filter: str) -> bool:
        """check if genotype matches the selected zygosity filter"""
        if zygosity_filter == 'All':
            return True
        if zygosity_filter == 'Homozygous':
            return GenotypeMatcher.is_homozygous(genotype)
        if zygosity_filter == 'Heterozygous':
            return not GenotypeMatcher.is_homozygous(genotype)
        return False

    def filter_by_ancestry(self, ancestry: str) -> pd.DataFrame:
        """filter dataframe by ancestry"""
        if ancestry != 'All':
            return self.df[self.df['ancestry'] == ancestry]
        return self.df

    def process_carriers(self, selected_variants: List[str], ancestry: str,
                         zygosity_filter: str) -> Optional[pd.DataFrame]:
        """process and return carrier data based on selected filters"""
        if not selected_variants:
            return None

        working_df = self.filter_by_ancestry(ancestry)
        carriers_status = []

        for _, row in working_df.iterrows():
            has_variant = any(GenotypeMatcher.is_carrier(row[variant])
                              for variant in selected_variants)

            if has_variant:
                status = self.get_carrier_status(
                    row, selected_variants, zygosity_filter)
                if status:
                    carriers_status.append({
                        'IID': row['IID'],
                        'ancestry': row['ancestry'],
                        'cohort': row['study'],
                        **status
                    })

        return pd.DataFrame(carriers_status) if carriers_status else None
