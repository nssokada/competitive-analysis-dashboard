
# ======================== data_loader.py ========================
"""Data loading and management"""

import json
import pandas as pd
import streamlit as st
from utils import DataProcessor

class DataLoader:
    """Handles data loading and caching"""
    
    def __init__(self):
        self.data = None
        self.programs_df = pd.DataFrame()
        self.companies_df = pd.DataFrame()
        self.trials_df_norm = pd.DataFrame()
        self.partnerships_df = pd.DataFrame()
        self.publications_df = pd.DataFrame()
        self.company_infos = {}
        
    def load_from_file(self, uploaded_file):
        """Load data from uploaded JSON file"""
        if uploaded_file is not None:
            content = uploaded_file.read()
            self.data = json.loads(content)['data']
            self._process_dataframes()
            return True
        return False
    
    def _process_dataframes(self):
        """Process raw data into dataframes"""
        if not self.data:
            return
            
        processor = DataProcessor()
        
        self.programs_df = pd.DataFrame(self.data.get('programs', []))
        self.companies_df = pd.DataFrame(self.data.get('companies', []))
        self.partnerships_df = pd.DataFrame(self.data.get('partnerships', []))
        self.publications_df = pd.DataFrame(self.data.get('publications', []))
        
        # Process trials with normalization
        trials_raw = pd.DataFrame(self.data.get('trials', []))
        self.trials_df_norm = processor.normalize_trials(trials_raw)
        
        # Load company info
        self.company_infos = processor.load_company_infos(self.data)
        
        # Add derived fields
        self._add_derived_fields()
    
    def _add_derived_fields(self):
        """Add calculated fields to dataframes"""
        if not self.programs_df.empty:
            # Add search blob for programs
            search_fields = [
                'program_name', 'company_name', 'target_primary',
                'indication_primary', 'mechanism_of_action_detailed_final',
                'biological_rationale_final'
            ]
            for field in search_fields:
                if field not in self.programs_df.columns:
                    self.programs_df[field] = ""
            
            self.programs_df['_search_blob'] = (
                self.programs_df[search_fields]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
            )
        
        if not self.companies_df.empty and not self.programs_df.empty:
            # Add active programs count
            prog_counts = self.programs_df['company_name'].value_counts()
            self.companies_df['active_programs'] = (
                self.companies_df['company_name']
                .map(prog_counts)
                .fillna(0)
                .astype(int)
            )
    
    def get_filtered_data(self, filters):
        """Apply filters and return filtered dataframes"""
        filtered = {'programs': self.programs_df.copy()}
        
        # Apply program filters
        if filters.get('indications'):
            filtered['programs'] = filtered['programs'][
                filtered['programs']['indication_group'].isin(filters['indications'])
            ]
        if filters.get('targets'):
            filtered['programs'] = filtered['programs'][
                filtered['programs']['target_family_final'].isin(filters['targets'])
            ]
        if filters.get('classifications'):
            filtered['programs'] = filtered['programs'][
                filtered['programs']['program_classification_final'].isin(filters['classifications'])
            ]
        if filters.get('modalities'):
            filtered['programs'] = filtered['programs'][
                filtered['programs']['modality_final'].isin(filters['modalities'])
            ]
        
        # Filter related data
        filtered_program_ids = set(filtered['programs']['program_id'].dropna().unique())
        filtered_company_names = set(filtered['programs']['company_name'].dropna().unique())
        
        filtered['trials'] = self.trials_df_norm[
            self.trials_df_norm['program_id'].isin(filtered_program_ids)
        ] if not self.trials_df_norm.empty else pd.DataFrame()
        
        filtered['partnerships'] = self._filter_partnerships(filtered_company_names)
        
        return filtered
    
    def _filter_partnerships(self, company_names):
        """Filter partnerships by company names"""
        if self.partnerships_df.empty:
            return pd.DataFrame()
        
        col_candidates = [
            c for c in self.partnerships_df.columns 
            if 'company' in c.lower() or 'partner' in c.lower()
        ]
        
        if not col_candidates:
            return pd.DataFrame()
        
        mask = False
        for col in col_candidates:
            mask = mask | self.partnerships_df[col].isin(company_names)
        
        return self.partnerships_df[mask]