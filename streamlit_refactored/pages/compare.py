# ======================== pages/compare.py ========================
"""Compare page"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pages.base import BasePage
from config import PlotlyTheme
from utils import Formatters

class ComparePage(BasePage):
    """Comparative Analysis page"""
    
    def _render_content(self):
        st.title("Comparative Analysis")
        
        programs_df = self.data_loader.programs_df
        companies_df = self.data_loader.companies_df
        trials_df = self.data_loader.trials_df_norm
        
        tab1, tab2 = st.tabs(["Portfolio Distribution", "Company Benchmarking"])
        
        with tab1:
            self._render_portfolio_distribution(programs_df)
        
        with tab2:
            self._render_company_benchmarking(companies_df, programs_df, trials_df)
    
    def _render_portfolio_distribution(self, programs_df):
        """Render portfolio distribution analysis"""
        st.markdown("### Portfolio Distribution")
        
        view = st.radio(
            "View",
            ["Company × Phase", "Cross-Dimensional Matrix"],
            horizontal=True
        )
        
        if view == "Company × Phase":
            self._render_company_phase_analysis(programs_df)
        else:
            self._render_cross_dimensional_analysis(programs_df)
    
    def _render_company_phase_analysis(self, programs_df):
        """Render company by phase analysis"""
        src = st.radio("Analyze:", ["Programs"], horizontal=True)
        min_items = st.number_input(
            "Min items per company (filter small players)",
            min_value=0,
            value=0,
            step=1
        )
        
        # Prepare data
        df = programs_df[['company_name', 'development_stage_final']].copy()
        df['company_name'] = df['company_name'].astype(str).str.strip()
        df['development_stage_final'] = df['development_stage_final'].astype(str).str.strip()
        
        # Phase order
        phase_order = [
            "Discovery", "Preclinical", "Phase 1", "Phase 2",
            "Phase 3", "Filed", "Approved"
        ]
        
        # Aggregate
        agg = (
            df.groupby(['company_name', 'development_stage_final'], dropna=False)
            .size()
            .reset_index(name='count')
        )
        
        if agg.empty:
            st.warning("No data to visualize.")
            return
        
        # Apply filters
        totals = (
            agg.groupby('company_name', as_index=False)['count']
            .sum()
            .sort_values('count', ascending=False)
        )
        
        if min_items > 0:
            keep = totals[totals['count'] >= min_items]['company_name']
            agg = agg[agg['company_name'].isin(keep)]
            totals = totals[totals['company_name'].isin(keep)]
        
        if totals.empty:
            st.info("All companies filtered out by the minimum threshold.")
            return
        
        # Top N selection
        top_default = min(10, len(totals))
        top_n = st.slider(
            "Show top N companies by total volume",
            3,
            min(50, len(totals)),
            top_default
        )
        top_companies = totals.head(top_n)['company_name'].tolist()
        agg = agg[agg['company_name'].isin(top_companies)]
        
        # Order categories
        agg['company_name'] = pd.Categorical(
            agg['company_name'],
            categories=top_companies,
            ordered=True
        )
        
        phase_cats = [p for p in phase_order if p in agg['development_stage_final'].unique()]
        if phase_cats:
            agg['development_stage_final'] = pd.Categorical(
                agg['development_stage_final'],
                categories=phase_cats + [
                    p for p in agg['development_stage_final'].unique()
                    if p not in phase_cats
                ],
                ordered=True
            )
        
        # Stacked bar chart
        fig_bar = px.bar(
            agg,
            x='company_name',
            y='count',
            color='development_stage_final',
            labels={
                'company_name': 'Company',
                'count': 'Count',
                'development_stage_final': 'Phase'
            },
            title="Programs by Phase per Company"
        )
        fig_bar.update_layout(
            **PlotlyTheme.LAYOUT,
            barmode="stack",
            xaxis_title=None,
            legend_title="Phase",
            height=480
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Heatmap
        pivot = agg.pivot_table(
            index='company_name',
            columns='development_stage_final',
            values='count',
            fill_value=0
        )
        pivot = pivot.reindex(index=top_companies)
        
        fig_heat = px.imshow(
            pivot,
            labels=dict(x="Phase", y="Company", color="Count"),
            title="Programs by Phase per Company (Heatmap)",
            aspect="auto"
        )
        fig_heat.update_layout(**PlotlyTheme.LAYOUT, height=520)
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Data table
        with st.expander("Show aggregated table"):
            st.dataframe(
                agg.sort_values(['company_name', 'development_stage_final']).rename(
                    columns={
                        'company_name': 'Company',
                        'development_stage_final': 'Phase',
                        'count': 'Count'
                    }
                ),
                use_container_width=True
            )
    
    def _render_cross_dimensional_analysis(self, programs_df):
        """Render cross-dimensional analysis"""
        st.markdown("### Cross-Dimensional Analysis")
        
        dimensions = [
            'indication_group',
            'target_family_final',
            'program_classification_final',
            'modality_final',
            'development_stage_final'
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            x_dim = st.selectbox("X-axis", dimensions, key="x_dim_matrix")
        with col2:
            y_dim = st.selectbox("Y-axis", dimensions, key="y_dim_matrix")
        
        if st.button("Generate Analysis", key="gen_matrix"):
            cross_tab = pd.crosstab(programs_df[y_dim], programs_df[x_dim])
            
            fig = px.imshow(
                cross_tab,
                labels=dict(x=x_dim, y=y_dim, color="Count"),
                title="Portfolio Distribution Matrix",
                aspect="auto",
                color_continuous_scale="Blues"
            )
            fig.update_layout(**PlotlyTheme.LAYOUT, height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_company_benchmarking(self, companies_df, programs_df, trials_df):
        """Render company benchmarking section"""
        st.markdown("### Company Benchmarking")
        
        company_names = sorted(
            companies_df['company_name'].dropna().astype(str).unique().tolist()
        )
        
        col1, col2 = st.columns([4, 2])
        with col1:
            selected = st.multiselect(
                "Select companies (up to 6)",
                options=company_names,
                default=company_names[:min(2, len(company_names))]
            )
        with col2:
            cols_per_row = st.slider("Columns", 2, 4, 3, step=1)
        
        if not selected:
            return
        
        if len(selected) > 6:
            st.warning("Showing the first 6 selected companies.")
            selected = selected[:6]
        
        # Render company cards in grid
        for start in range(0, len(selected), cols_per_row):
            row = selected[start:start + cols_per_row]
            cols = st.columns(len(row))
            for i, company_name in enumerate(row):
                with cols[i]:
                    self._render_benchmark_card(
                        company_name,
                        companies_df,
                        programs_df,
                        trials_df
                    )
    
    def _render_benchmark_card(self, company_name, companies_df, programs_df, trials_df):
        """Render individual company benchmark card"""
        # Get company data
        company_mask = companies_df['company_name'].str.lower() == company_name.lower()
        company_data = companies_df[company_mask]
        
        if company_data.empty:
            st.warning(f"No data found for {company_name}")
            return
        
        company_dict = company_data.iloc[0].to_dict()
        
        # Get related data
        progs = programs_df[
            programs_df['company_name'].str.lower() == company_name.lower()
        ] if not programs_df.empty else pd.DataFrame()
        
        trials = trials_df[
            trials_df['company_name'].str.lower() == company_name.lower()
        ] if not trials_df.empty else pd.DataFrame()
        
        # Render card
        title = str(company_name).title()
        if len(title) > 48:
            title = title[:45] + "…"
        st.markdown(f"## {title}")
        
        with st.expander("Key Metrics", expanded=False):
            st.markdown(f"**Type:** {Formatters.safe_get(company_dict, 'public_private')}")
            st.markdown(f"**Founded:** {Formatters.safe_get(company_dict, 'founding_year')}")
            st.markdown(f"**Total Funding:** {Formatters.format_currency(company_dict.get('total_funding_numeric', 0))}")
            st.markdown(f"**Active Programs:** {len(progs)}")
        
        with st.expander("Company Details", expanded=False):
            st.markdown(f"**Country:** {Formatters.safe_get(company_dict, 'country_normalized')}")
            st.markdown(f"**HQ City:** {Formatters.safe_get(company_dict, 'city_normalized')}")
            st.markdown(f"**Website:** {Formatters.safe_get(company_dict, 'website')}")
            st.markdown(f"**Size:** {Formatters.safe_get(company_dict, 'size_category')}")
            st.markdown(f"**Primary Focus:** {Formatters.safe_get(company_dict, 'company_focus')}")
            st.markdown(f"**Leadership:** {Formatters.safe_get(company_dict, 'leadership')}")
        
        with st.expander("Programs", expanded=False):
            if progs.empty:
                st.info("No programs found.")
            else:
                prog_cols = [
                    c for c in [
                        'program_name', 'development_stage_final',
                        'indication_primary', 'program_classification_final',
                        'modality_final', 'platform_delivery_final',
                        'target_primary'
                    ] if c in progs.columns
                ]
                st.dataframe(progs[prog_cols], use_container_width=True, hide_index=True)
        
        with st.expander("Clinical Trials", expanded=False):
            if trials.empty:
                st.info("No trials found.")
            else:
                view = trials.drop_duplicates(subset=['trial_key'])
                trial_cols = [
                    c for c in [
                        'trial_id', 'phase', 'status', 'indication',
                        'enrollment_target', 'program_name', 'nct_id'
                    ] if c in view.columns
                ]
                st.dataframe(view[trial_cols], use_container_width=True, hide_index=True)
        
        with st.expander("Notes", expanded=False):
            st.markdown(f"**Research Notes:** {Formatters.safe_get(company_dict, 'research_notes')}")