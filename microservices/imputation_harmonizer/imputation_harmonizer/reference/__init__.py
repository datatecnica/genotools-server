"""Reference panel loaders for HRC, 1000G, and TOPMed."""

from imputation_harmonizer.reference.base import ReferencePanel
from imputation_harmonizer.reference.hrc import HRCPanel
from imputation_harmonizer.reference.kg import KGPanel
from imputation_harmonizer.reference.topmed import TOPMedPanel

__all__ = ["ReferencePanel", "HRCPanel", "KGPanel", "TOPMedPanel"]
