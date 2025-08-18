

# ======================== utils.py ========================
"""Utility functions and data processors"""

import pandas as pd
import numpy as np
import ast
import re
from typing import List, Dict, Any, Optional

class DataProcessor:
    """Data processing utilities"""
    
    @staticmethod
    def normalize_trials(trials_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and expand trial data"""
        if trials_df is None or trials_df.empty:
            return pd.DataFrame()
        
        df = trials_df.copy()
        
        # Process NCT IDs
        df['__nct_list'] = df['trial_id'].apply(DataProcessor.parse_nct_ids)
        df['__nct_list'] = df['__nct_list'].apply(lambda x: x if x else [None])
        
        # Explode to one row per NCT
        df = df.explode('__nct_list', ignore_index=True)
        df['nct_id'] = df['__nct_list']
        df.drop(columns=['__nct_list'], inplace=True)
        
        # Create stable trial key
        df['trial_key'] = df.apply(DataProcessor._create_trial_key, axis=1)
        df['trial_label'] = np.where(
            df['nct_id'].notna() & (df['nct_id'].astype(str).str.len() > 0),
            df['nct_id'],
            df['trial_key']
        )
        
        # Process multi-value fields
        df = DataProcessor._process_trial_fields(df)
        
        return df
    
    @staticmethod
    def _create_trial_key(row):
        """Create a stable trial key"""
        if pd.notna(row.get('nct_id')) and str(row['nct_id']).strip():
            return str(row['nct_id'])
        
        parts = [
            str(row.get('program_id', 'NA')),
            str(row.get('phase', 'NA')),
            str(row.get('trial_title', 'NA'))[:120]
        ]
        return "||".join(parts)
    
    @staticmethod
    def _process_trial_fields(df):
        """Process multi-value trial fields"""
        # Phase
        if 'phase' in df.columns:
            df['phase_list'] = df['phase'].apply(DataProcessor.parse_listish)
            df['phase_clean'] = df['phase_list'].apply(DataProcessor.canonical_phase)
        else:
            df['phase_list'] = [[] for _ in range(len(df))]
            df['phase_clean'] = ""
        
        # Status
        if 'status' in df.columns:
            df['status_list'] = df['status'].apply(DataProcessor.parse_listish)
            df['status_clean'] = df['status_list'].apply(
                lambda xs: " / ".join(DataProcessor.unique_preserve(xs))
            )
        else:
            df['status_list'] = [[] for _ in range(len(df))]
            df['status_clean'] = ""
        
        # Countries
        if 'countries_normalized' in df.columns:
            df['countries_list'] = df['countries_normalized'].apply(DataProcessor.parse_listish)
            df['countries_clean'] = df['countries_list'].apply(
                lambda xs: " • ".join(DataProcessor.unique_preserve(xs))
            )
        else:
            df['countries_list'] = [[] for _ in range(len(df))]
            df['countries_clean'] = ""
        
        # Title
        if 'trial_title' in df.columns:
            df['title_list'] = df['trial_title'].apply(DataProcessor.parse_listish)
            df['trial_title_clean'] = df.apply(
                lambda r: (r['title_list'][0] if r['title_list'] 
                          else (str(r['trial_title']) if pd.notna(r['trial_title']) else "")),
                axis=1
            )
        else:
            df['title_list'] = [[] for _ in range(len(df))]
            df['trial_title_clean'] = ""
        
        return df
    
    @staticmethod
    def parse_listish(raw):
        """Parse various list-like formats into clean Python list"""
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return []
        if isinstance(raw, (list, tuple)):
            return [str(x).strip() for x in raw if str(x).strip()]
        
        s = str(raw).strip()
        if not s:
            return []
        
        # Try to parse as Python literal
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        
        # Fallback: split on delimiters
        parts = re.split(r"[|;/]+", s)
        return [p.strip() for p in parts if p.strip()]
    
    @staticmethod
    def parse_nct_ids(raw):
        """Extract NCT IDs from various formats"""
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return []
        
        ids = []
        
        # Handle list/tuple
        if isinstance(raw, (list, tuple)):
            candidates = [str(x) for x in raw]
        else:
            s = str(raw).strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        candidates = [str(x) for x in parsed]
                    else:
                        candidates = [s]
                except Exception:
                    candidates = [s]
            else:
                candidates = [s]
        
        # Extract NCT IDs
        for c in candidates:
            for m in re.findall(r"NCT\d{8}", c):
                if m not in ids:
                    ids.append(m)
        
        return ids
    
    @staticmethod
    def unique_preserve(seq):
        """Remove duplicates while preserving order"""
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    
    @staticmethod
    def canonical_phase(ph_list):
        """Convert phase list to canonical string"""
        phase_rank = {
            "Preclinical": 0,
            "Phase 1": 1,
            "Phase 1/2": 2,
            "Phase 2": 3,
            "Phase 2/3": 4,
            "Phase 3": 5,
            "Filed": 6,
            "Approved": 7
        }
        
        if not ph_list:
            return ""
        
        ph_uniq = DataProcessor.unique_preserve(ph_list)
        ph_sorted = sorted(ph_uniq, key=lambda p: phase_rank.get(p, -1))
        return " / ".join(ph_sorted)
    
    @staticmethod
    def load_company_infos(data: dict) -> dict:
        """Load company extra information"""
        infos = {}
        
        # Check various possible keys
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
        
        return infos

class Formatters:
    """Formatting utilities"""
    
    @staticmethod
    def format_currency(value):
        """Format currency values"""
        if pd.isna(value) or value == 0:
            return "Undisclosed"
        return f"${value/1e6:.1f}M"
    
    @staticmethod
    def safe_get(dict_obj, key, default="N/A"):
        """Safely extract and format display string"""
        value = dict_obj.get(key, default)
        
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            return "\n".join(str(item) for item in value)
        
        if isinstance(value, str):
            s = value.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        if not parsed:
                            return default
                        return "\n".join(str(item) for item in parsed)
                except (ValueError, SyntaxError):
                    inner = s[1:-1]
                    parts = [
                        part.strip().strip("'\"")
                        for part in inner.split(",")
                        if part.strip().strip("'\"")
                    ]
                    if parts:
                        return "\n".join(parts)
                    else:
                        return default
        
        return str(value)
    
    @staticmethod
    def trial_links(raw):
        """Create markdown links for NCT IDs"""
        ids = DataProcessor.parse_nct_ids(raw)
        if not ids:
            return "N/A"
        return " • ".join(
            f"[{tid}](https://clinicaltrials.gov/study/{tid})" 
            for tid in ids
        )
