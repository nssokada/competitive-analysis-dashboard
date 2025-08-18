# ======================== pages/clinical_trials.py ========================
"""Clinical Trials page"""

import streamlit as st
import pandas as pd
from pages.base import BasePage
from utils import Formatters

class ClinicalTrialsPage(BasePage):
    """Clinical Trial Analysis page"""
    
    def _render_content(self):
        st.title("Clinical Trial Analysis")
        
        trials_df = self.data_loader.trials_df_norm
        
        if trials_df.empty:
            st.info("No trial data available.")
            return
        
        # Apply filters
        filtered_trials = self._apply_filters(trials_df)
        
        # Display results
        dedup = filtered_trials.drop_duplicates(subset=['trial_key'])
        st.markdown(f"### Results: {len(dedup)} trials")
        
        # Display trials
        for _, trial in dedup.iterrows():
            self._render_trial_card(trial)
    
    def _apply_filters(self, trials_df):
        """Apply filters to trials dataframe"""
        # Build filter options
        phase_options = ['All']
        if 'phase_list' in trials_df.columns:
            phase_atomic = sorted({p for lst in trials_df['phase_list'] for p in (lst or [])})
            phase_options += phase_atomic
        
        status_options = ['All']
        if 'status_list' in trials_df.columns:
            status_atomic = sorted({s for lst in trials_df['status_list'] for s in (lst or [])})
            status_options += status_atomic
        
        indication_options = (
            ['All'] + sorted(trials_df['indication'].dropna().unique())
            if 'indication' in trials_df.columns
            else ['All']
        )
        
        company_options = (
            ['All'] + sorted(trials_df['company_name'].dropna().unique())
            if 'company_name' in trials_df.columns
            else ['All']
        )
        
        # Render filter controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            selected_indication = st.selectbox("Indication", indication_options)
        with col2:
            selected_phase = st.selectbox("Phase", phase_options)
        with col3:
            selected_status = st.selectbox("Status", status_options)
        with col4:
            selected_company = st.selectbox("Sponsor", company_options)
        
        # Apply filters
        filtered = trials_df.copy()
        
        if selected_indication != 'All' and 'indication' in filtered.columns:
            filtered = filtered[filtered['indication'] == selected_indication]
        
        if selected_phase != 'All' and 'phase_list' in filtered.columns:
            filtered = filtered[
                filtered['phase_list'].apply(lambda L: selected_phase in (L or []))
            ]
        
        if selected_status != 'All' and 'status_list' in filtered.columns:
            filtered = filtered[
                filtered['status_list'].apply(lambda L: selected_status in (L or []))
            ]
        
        if selected_company != 'All' and 'company_name' in filtered.columns:
            filtered = filtered[filtered['company_name'] == selected_company]
        
        return filtered
    
    def _render_trial_card(self, trial):
        """Render individual trial card"""
        title_left = (
            trial['nct_id']
            if pd.notna(trial['nct_id']) and trial['nct_id']
            else trial['trial_label']
        )
        sponsor_disp = (
            trial['company_name'].title()
            if pd.notna(trial.get('company_name'))
            else 'Unknown'
        )
        
        with st.expander(f"{title_left} - {sponsor_disp}"):
            link_source = trial.get('nct_id') or trial.get('trial_id')
            st.markdown(f"**ClinicalTrials.gov:** {Formatters.trial_links(link_source)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Program:** {Formatters.safe_get(trial, 'program_name')}")
                st.markdown(f"**Status:** {trial.get('status_clean', '')}")
                st.markdown(f"**Indication:** {Formatters.safe_get(trial, 'indication')}")
            
            with col2:
                st.markdown(f"**Sponsor:** {Formatters.safe_get(trial, 'sponsor')}")
                st.markdown(f"**Countries:** {trial.get('countries_clean', '')}")
                st.markdown(f"**Title:** {trial.get('trial_title_clean', '')}")