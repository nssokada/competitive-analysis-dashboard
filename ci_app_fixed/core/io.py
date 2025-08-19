
from __future__ import annotations
import json, os, glob
import pandas as pd
import numpy as np
import streamlit as st
from .utils import parse_listish, unique_preserve, canonical_phase, parse_nct_ids, norm_str
from .utils import safe_get  # re-export for convenience

@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes: bytes | None):
    """Load and parse uploaded JSON bytes; return payload or None."""
    if uploaded_bytes is None:
        return None
    data = json.loads(uploaded_bytes)
    return data.get("data") or data

@st.cache_data(show_spinner=False)
def normalize_trials(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Explode multi-NCT rows into one row per trial and assign a stable trial_key."""
    if trials_df is None or trials_df.empty:
        return pd.DataFrame()

    df = trials_df.copy()
    if 'trial_id' not in df.columns:
        df['trial_id'] = None
    df['__nct_list'] = df['trial_id'].apply(parse_nct_ids)
    df['__nct_list'] = df['__nct_list'].apply(lambda x: x if x else [None])
    df = df.explode('__nct_list', ignore_index=True)
    df['nct_id'] = df['__nct_list']
    df.drop(columns=['__nct_list'], inplace=True)

    def _fallback_key(r):
        parts = [str(r.get('program_id','NA')), str(r.get('phase','NA')), str(r.get('trial_title','NA'))[:120]]
        return "||".join(parts)

    df['trial_key'] = np.where(
        df['nct_id'].notna() & (df['nct_id'].astype(str).str.len() > 0),
        df['nct_id'].astype(str),
        df.apply(_fallback_key, axis=1)
    )
    df['trial_label'] = np.where(
        df['nct_id'].notna() & (df['nct_id'].astype(str).str.len() > 0),
        df['nct_id'],
        df['trial_key']
    )

    # Clean multi-value fields
    if 'phase' in df.columns:
        df['phase_list'] = df['phase'].apply(parse_listish)
        df['phase_clean'] = df['phase_list'].apply(canonical_phase)
    else:
        df['phase_list'] = [[] for _ in range(len(df))]
        df['phase_clean'] = ""

    if 'status' in df.columns:
        df['status_list'] = df['status'].apply(parse_listish)
        df['status_clean'] = df['status_list'].apply(lambda xs: " / ".join(unique_preserve(xs)))
    else:
        df['status_list'] = [[] for _ in range(len(df))]
        df['status_clean'] = ""

    if 'countries_normalized' in df.columns:
        df['countries_list'] = df['countries_normalized'].apply(parse_listish)
        df['countries_clean'] = df['countries_list'].apply(lambda xs: " • ".join(unique_preserve(xs)))
    else:
        df['countries_list'] = [[] for _ in range(len(df))]
        df['countries_clean'] = ""

    if 'trial_title' in df.columns:
        def _pick_title(r):
            lst = parse_listish(r.get('trial_title'))
            if lst:
                return lst[0]
            v = r.get('trial_title')
            return str(v) if pd.notna(v) else ""
        df['trial_title_clean'] = df.apply(_pick_title, axis=1)
    else:
        df['trial_title_clean'] = ""

    return df

def load_company_infos_blob(data: dict, extra_dir: str | None = None) -> dict:
    """
    Returns {company_id: info_dict}. Sources:
    1) Top-level keys in uploaded data: 'company_infos' or variants
    2) Optional local directory of JSON files named COMP_XXXX.json
    """
    infos: dict[str, dict] = {}
    for key in ["company_infos", "company_info", "companies_extra", "companies_info"]:
        if key in data and data[key]:
            raw = data[key]
            if isinstance(raw, dict):
                for cid, obj in raw.items():
                    infos[str(cid)] = obj
            elif isinstance(raw, list):
                for obj in raw:
                    cid = str(obj.get("company_id") or obj.get("id") or "")
                    if cid:
                        infos[cid] = obj
    if extra_dir:
        try:
            for p in glob.glob(os.path.join(extra_dir, "*.json")):
                with open(p, "r") as f:
                    obj = json.load(f)
                cid = str(obj.get("company_id") or os.path.splitext(os.path.basename(p))[0])
                infos[cid] = obj
        except Exception:
            pass
    return infos

def build_search_blob(programs_df: pd.DataFrame) -> pd.DataFrame:
    fields = [
        'program_name','company_name','target_primary','indication_primary',
        'mechanism_of_action_detailed_final','biological_rationale_final'
    ]
    for c in fields:
        if c not in programs_df.columns:
            programs_df[c] = ""
    programs_df['_search_blob'] = (
        programs_df[fields]
        .fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    )
    return programs_df
