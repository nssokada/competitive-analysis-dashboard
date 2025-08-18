
# ======================== pages/overview.py ========================
"""Overview page"""

import streamlit as st
import plotly.express as px
from pages.base import BasePage
from components import MetricsRow, FilterBar
from config import PlotlyTheme

class OverviewPage(BasePage):
    """Overview page with data upload and summary statistics"""
    
    def _render_content(self):
        st.title("Overview")
        
        # Data import section
        self._render_data_import()
        
        if not self.data_loader or not self.data_loader.data:
            st.info("Please upload a dataset to begin analysis")
            return
        
        # Filters and metrics
        self._render_filters_and_metrics()
        
        # Visualizations
        self._render_visualizations()
    
    def _render_data_import(self):
        """Render data import section"""
        st.markdown("### Data Import")
        uploaded_file = st.file_uploader(
            "Select JSON dataset",
            type=['json', 'txt'],
            help="Upload competitive intelligence dataset in JSON format"
        )
        
        if uploaded_file is not None:
            if self.data_loader.load_from_file(uploaded_file):
                st.success("Data loaded successfully")
                st.session_state.data_loader = self.data_loader
    
    def _render_filters_and_metrics(self):
        """Render filters and key metrics"""
        st.markdown("### Filters")
        
        # Setup filter configuration
        filter_config = [
            ('indications', 'indication_group', 'Indication'),
            ('targets', 'target_family_final', 'Target Family'),
            ('classifications', 'program_classification_final', 'Classification'),
            ('modalities', 'modality_final', 'Modality')
        ]
        
        filters = FilterBar.render_multiselect(
            self.data_loader.programs_df,
            filter_config
        )
        
        # Get filtered data
        filtered_data = self.data_loader.get_filtered_data(filters)
        
        # Display metrics
        st.markdown("### Key Metrics")
        metrics = {
            "Total Companies": len(set(filtered_data['programs']['company_name'].dropna().unique())),
            "Active Programs": len(filtered_data['programs']),
            "Clinical Trials": len(filtered_data['trials'].drop_duplicates(subset=['trial_key'])) if not filtered_data['trials'].empty else 0,
            "Partnerships": len(filtered_data['partnerships'])
        }
        MetricsRow.render(metrics)
        
        # Store filtered data for visualizations
        self.filtered_data = filtered_data
    
    def _render_visualizations(self):
        """Render overview visualizations"""
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        programs_df = self.filtered_data['programs']
        
        # Therapeutic area distribution
        st.markdown("### Therapeutic Area Distribution")
        if 'indication_group' in programs_df.columns and not programs_df.empty:
            ind_counts = programs_df['indication_group'].value_counts().head(10)
            if not ind_counts.empty:
                fig = px.bar(
                    x=ind_counts.values,
                    y=ind_counts.index,
                    orientation='h',
                    labels={'x': 'Number of Programs', 'y': 'Indication'},
                    title="Leading Therapeutic Areas (filtered)"
                )
                fig.update_layout(**PlotlyTheme.LAYOUT, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Stage and modality charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'development_stage_final' in programs_df.columns and not programs_df.empty:
                stage_counts = programs_df['development_stage_final'].value_counts()
                if not stage_counts.empty:
                    fig = px.pie(
                        values=stage_counts.values,
                        names=stage_counts.index,
                        title="Development Stage Distribution",
                        hole=0.4
                    )
                    fig.update_layout(**PlotlyTheme.LAYOUT)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'modality_final' in programs_df.columns and not programs_df.empty:
                mod_counts = programs_df['modality_final'].value_counts().head(8)
                if not mod_counts.empty:
                    fig = px.bar(
                        x=mod_counts.values,
                        y=mod_counts.index,
                        orientation='h',
                        title="Technology Platforms"
                    )
                    fig.update_layout(**PlotlyTheme.LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)