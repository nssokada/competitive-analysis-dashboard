# ======================== pages/companies.py ========================
"""Companies page"""

import streamlit as st
import pandas as pd
import numpy as np
from pages.base import BasePage
from components import MetricsRow, FilterBar
from utils import Formatters, DataProcessor
from typing import Dict, Any

class CompaniesPage(BasePage):
    """Companies analysis page"""
    
    def _render_content(self):
        st.title("Companies")
        
        companies_df = self.data_loader.companies_df
        programs_df = self.data_loader.programs_df
        trials_df = self.data_loader.trials_df_norm
        publications_df = self.data_loader.publications_df
        company_infos = self.data_loader.company_infos
        
        st.caption(f"Loaded extras for {len(company_infos)} companies")
        
        if companies_df.empty:
            st.info("No company data available.")
            return
        
        # Apply filters
        filtered_df = self._apply_filters(companies_df)
        
        # Display results
        st.markdown(f"### Results: {len(filtered_df)} companies")
        
        # Render company cards
        for _, company in filtered_df.iterrows():
            self._render_company_card(
                company,
                programs_df,
                trials_df,
                publications_df,
                company_infos
            )
    
    def _apply_filters(self, df):
        """Apply filters to companies dataframe"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            name_query = st.text_input("Search name", "")
        with col2:
            types = ['All'] + sorted(df['public_private'].dropna().unique())
            sel_type = st.selectbox("Type", types)
        with col3:
            countries = ['All'] + sorted(df['country_normalized'].dropna().unique())
            sel_country = st.selectbox("Country", countries)
        with col4:
            sizes = ['All'] + sorted(df['size_category'].dropna().unique())
            sel_size = st.selectbox("Size", sizes)
        
        sort_by = st.selectbox(
            "Sort by",
            ["Name (A→Z)", "Active programs (↓)", "Total funding (↓)"]
        )
        
        # Apply filters
        filtered = df.copy()
        
        if name_query.strip():
            filtered = filtered[
                filtered['company_name'].fillna("").str.lower().str.contains(
                    name_query.strip().lower()
                )
            ]
        
        if sel_type != 'All':
            filtered = filtered[filtered['public_private'] == sel_type]
        
        if sel_country != 'All':
            filtered = filtered[filtered['country_normalized'] == sel_country]
        
        if sel_size != 'All':
            filtered = filtered[filtered['size_category'] == sel_size]
        
        # Apply sorting
        if sort_by == "Name (A→Z)":
            filtered = filtered.sort_values('company_name', ascending=True, na_position='last')
        elif sort_by == "Active programs (↓)":
            filtered = filtered.sort_values('active_programs', ascending=False, na_position='last')
        else:
            if 'total_funding_numeric' in filtered.columns:
                filtered = filtered.sort_values('total_funding_numeric', ascending=False, na_position='last')
        
        return filtered
    
    def _render_company_card(self, company, programs_df, trials_df, publications_df, company_infos):
        """Render individual company card"""
        name = company.get('company_name', 'Unknown')
        active_programs = int(company.get('active_programs', 0))
        
        headline = (
            f"{str(name).title()} — {Formatters.safe_get(company, 'public_private')} | "
            f"{Formatters.safe_get(company, 'country_normalized')} | {active_programs} programs"
        )
        
        with st.expander(headline):
            # Company Information
            st.markdown("#### Company Information")
            
            cols = st.columns(4)
            with cols[0]:
                st.markdown(f"**Type:** {Formatters.safe_get(company, 'public_private')}")
            with cols[1]:
                st.markdown(f"**Country:** {Formatters.safe_get(company, 'country_normalized')}")
            with cols[2]:
                st.markdown(f"**Size:** {Formatters.safe_get(company, 'size_category')}")
            with cols[3]:
                st.markdown(f"**Founded:** {Formatters.safe_get(company, 'founding_year')}")
            
            # Metrics
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                funding = company.get('total_funding_numeric', np.nan)
                st.metric("Total Funding", Formatters.format_currency(funding if pd.notna(funding) else 0))
            with metrics_cols[1]:
                st.metric("Active Programs", active_programs)
            with metrics_cols[2]:
                st.metric("HQ City", Formatters.safe_get(company, 'city_normalized'))
            with metrics_cols[3]:
                st.markdown(
                    f"<div style='font-size: 0.85rem; line-height: 1.2em;'>"
                    f"<b>Primary Focus:</b><br>{Formatters.safe_get(company, 'company_focus')}</div>",
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # Programs section
            self._render_company_programs(name, programs_df, trials_df, publications_df)
            
            # Company-level trials
            self._render_company_trials(name, trials_df)
            
            # Additional information
            st.markdown("#### Additional Information")
            st.markdown(f"**Leadership:** {Formatters.safe_get(company, 'leadership')}")
            st.markdown(f"**Website:** {Formatters.safe_get(company, 'website')}")
            st.markdown(f"**Research Notes:** {Formatters.safe_get(company, 'research_notes')}")
            
            # Company intelligence (funding, partnerships, etc.)
            self._render_company_extras(company.to_dict(), company_infos)
    
    def _render_company_programs(self, company_name, programs_df, trials_df, publications_df):
        """Render programs for a company"""
        st.markdown("#### Programs")
        
        comp_programs = programs_df[programs_df['company_name'] == company_name]
        
        if comp_programs.empty:
            st.info("No programs found for this company.")
            return
        
        for _, program in comp_programs.iterrows():
            with st.expander(
                f"{program['program_name'].upper()} | {program['development_stage_final']}"
            ):
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Program:** {program['program_name']}")
                    st.markdown(f"**Classification:** {program['program_classification_final']}")
                    st.markdown(f"**Target:** {program['target_primary']}")
                with cols[1]:
                    st.markdown(f"**Indication:** {program['indication_primary']}")
                    st.markdown(f"**Delivery:** {program['platform_delivery_final']}")
                    st.markdown(f"**Stage:** {program['development_stage_final']}")
                
                st.markdown(f"**Biological Rationale:** {Formatters.safe_get(program, 'biological_rationale_final')}")
                st.markdown(f"**Mechanism of Action:** {Formatters.safe_get(program, 'mechanism_of_action_detailed_final')}")
                
                # Clinical trials for this program
                if not trials_df.empty:
                    prog_trials = trials_df[trials_df['program_id'] == program['program_id']]
                    if not prog_trials.empty:
                        st.markdown("**Clinical Development:**")
                        for _, trial in prog_trials.drop_duplicates(subset=['trial_key']).iterrows():
                            nct_display = (
                                f"[{trial['nct_id']}](https://clinicaltrials.gov/study/{trial['nct_id']})"
                                if pd.notna(trial['nct_id']) and trial['nct_id']
                                else "N/A"
                            )
                            st.markdown(f"- **{nct_display}**: {Formatters.safe_get(trial, 'phase')} – {Formatters.safe_get(trial, 'status')}")
                
                # Risk factors
                red_flags = Formatters.safe_get(program, 'red_flags')
                if red_flags != 'N/A':
                    st.warning(f"**Risk Factors:** {red_flags}")
    
    def _render_company_trials(self, company_name, trials_df):
        """Render company-level trials"""
        st.markdown("#### Clinical Trials (Company Level)")
        
        if trials_df.empty:
            st.info("No company-level trials found.")
            return
        
        comp_trials = trials_df[trials_df['company_name'] == company_name]
        
        if comp_trials.empty:
            st.info("No company-level trials found.")
            return
        
        for _, trial in comp_trials.drop_duplicates(subset=['trial_key']).iterrows():
            nct_display = (
                f"[{trial['nct_id']}](https://clinicaltrials.gov/study/{trial['nct_id']})"
                if pd.notna(trial['nct_id']) and trial['nct_id']
                else "N/A"
            )
            st.markdown(f"- **{nct_display}**: {Formatters.safe_get(trial, 'phase')} – {Formatters.safe_get(trial, 'status')}")
    
    def _render_company_extras(self, company_dict, company_infos):
        """Render company extras (funding, partnerships, etc.)"""
        if not company_infos:
            return
        
        # Implementation would go here for rendering funding, partnerships, M&A, etc.
        # This is a placeholder - you can add the full implementation based on your needs
        pass