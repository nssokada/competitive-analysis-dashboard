# ======================== components.py ========================
"""Reusable UI components"""

import streamlit as st
import pandas as pd
from utils import Formatters

class Sidebar:
    """Sidebar navigation component"""
    
    def __init__(self):
        self.pages = [
            'Overview', 
            'Programs', 
            'Companies', 
            'Clinical Trials', 
            'Compare', 
            'Cluster Analysis'
        ]
    
    def render(self):
        """Render sidebar and return selected page"""
        with st.sidebar:
            st.markdown("## COMPETITIVE INTELLIGENCE")
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            selected_page = st.radio("Navigation", self.pages, key='nav_radio')
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.caption("Competitive Intelligence Viz v2.0")
            
        return selected_page

class MetricsRow:
    """Component for displaying metrics in a row"""
    
    @staticmethod
    def render(metrics: dict):
        """Render a row of metrics"""
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics.items()):
            with col:
                st.metric(label, value)

class ProgramCard:
    """Component for displaying program information"""
    
    @staticmethod
    def render(program, company=None, trials_df=None, publications_df=None):
        """Render program information card"""
        with st.expander(
            f"{program['program_name'].upper()} - {program['company_name'].title()} | "
            f"{program['development_stage_final']}"
        ):
            # Company info if available
            if company:
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
                st.markdown("---")
            
            # Program details
            st.markdown("#### Program Details")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Program:** {program['program_name']}")
                st.markdown(f"**Classification:** {program['program_classification_final']}")
                st.markdown(f"**Target:** {program['target_primary']}")
            with c2:
                st.markdown(f"**Indication:** {program['indication_primary']}")
                st.markdown(f"**Delivery:** {program['platform_delivery_final']}")
                st.markdown(f"**Stage:** {program['development_stage_final']}")
            
            # Scientific details
            st.markdown("#### Scientific Rationale")
            st.markdown(f"**Biological Rationale:** {Formatters.safe_get(program, 'biological_rationale_final')}")
            st.markdown(f"**Mechanism of Action:** {Formatters.safe_get(program, 'mechanism_of_action_detailed_final')}")
            
            # Clinical trials if available
            if trials_df is not None and not trials_df.empty:
                prog_trials = trials_df[trials_df['program_id'] == program['program_id']]
                if not prog_trials.empty:
                    st.markdown("#### Clinical Development")
                    prog_trials = prog_trials.drop_duplicates(subset=['trial_key'])
                    for _, tr in prog_trials.iterrows():
                        if pd.notna(tr['nct_id']) and tr['nct_id']:
                            nct_display = f"[{tr['nct_id']}](https://clinicaltrials.gov/study/{tr['nct_id']})"
                        else:
                            nct_display = "N/A"
                        st.markdown(f"- **{nct_display}**: {Formatters.safe_get(tr, 'phase')} – {Formatters.safe_get(tr, 'status')}")
            
            # Additional info
            st.markdown("#### Additional Information")
            st.markdown(f"**Research Notes:** {Formatters.safe_get(program, 'research_notes')}")
            st.markdown(f"**Key Publications:** {Formatters.safe_get(program, 'key_scientific_paper')}")
            st.markdown(f"**Data Quality Index:** {Formatters.safe_get(program, 'data_quality_index')}")
            
            # Risk factors
            red_flags = Formatters.safe_get(program, 'red_flags')
            if red_flags != 'N/A':
                st.warning(f"**Risk Factors:** {red_flags}")

class FilterBar:
    """Component for filter controls"""
    
    @staticmethod
    def render_multiselect(df, columns_config):
        """Render multiselect filters"""
        filters = {}
        cols = st.columns(len(columns_config))
        
        for col, (key, column, label) in zip(cols, columns_config):
            with col:
                if column in df.columns:
                    options = sorted([x for x in df[column].dropna().unique() if str(x).strip() != ""])
                    filters[key] = st.multiselect(label, options=options, default=[])
                else:
                    filters[key] = []
        
        return filters
    
    @staticmethod
    def render_selectbox(df, columns_config):
        """Render selectbox filters"""
        filters = {}
        cols = st.columns(len(columns_config))
        
        for col, (key, column, label) in zip(cols, columns_config):
            with col:
                if column in df.columns:
                    options = ['All'] + sorted([x for x in df[column].dropna().unique() if str(x).strip() != ""])
                    filters[key] = st.selectbox(label, options)
                else:
                    filters[key] = 'All'
        
        return filters
