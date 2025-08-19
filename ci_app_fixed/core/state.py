
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

# Session keys
DATA = "data"
CURRENT_PAGE = "current_page"
TRIALS_DF_NORM = "trials_df_norm"
PUBLICATIONS_DF = "publications_df"
COMPANY_INFOS = "company_infos"

DEFAULT_PAGE = "Overview"

@dataclass
class SessionData:
    data: dict | None = None
    trials_df_norm: pd.DataFrame = field(default_factory=pd.DataFrame)
    publications_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    company_infos: dict = field(default_factory=dict)
