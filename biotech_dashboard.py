import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from datetime import datetime
import networkx as nx
from itertools import combinations
import ast
import re


# Page config
st.set_page_config(
    page_title="Competitive Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional dark theme CSS with better contrast
st.markdown("""
<style>
  /* only override non-form widgets now: */
  div[data-testid="metric-container"] { 
    background-color: #252830;
    border: 1px solid #3a3f4b;
    /* …etc… */
  }
  .streamlit-expanderHeader { /* … */ }
  .streamlit-expanderContent { /* … */ }
  /* you no longer need any .stSelectbox or popover rules */
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Overview'

def load_data(uploaded_file):
    """Load and parse the uploaded JSON data"""
    if uploaded_file is not None:
        content = uploaded_file.read()
        data = json.loads(content)
        return data['data']
    return None

def format_currency(value):
    """Format currency values"""
    if pd.isna(value) or value == 0:
        return "Undisclosed"
    return f"${value/1e6:.1f}M"

def parse_nct_ids(raw):
    """Return a de-duplicated, order‐preserving list of NCT IDs from messy inputs."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    ids = []

    # If it's already a list/tuple
    if isinstance(raw, (list, tuple)):
        candidates = [str(x) for x in raw]
    else:
        s = str(raw).strip()
        # If it's a string that looks like a Python list, try to parse
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

    # From all candidates, extract NCT IDs by regex
    for c in candidates:
        for m in re.findall(r"NCT\d{8}", c):
            if m not in ids:
                ids.append(m)
    return ids

def trial_links(raw):
    """Markdown links to ClinicalTrials.gov for one or more NCT IDs."""
    ids = parse_nct_ids(raw)
    if not ids:
        return "N/A"
    return " • ".join(f"[{tid}](https://clinicaltrials.gov/study/{tid})" for tid in ids)



def safe_get(dict_obj, key, default="N/A"):
    """
    Safely extract a display string:
      • None / NaN / empty list    → default
      • real list/tuple             → "\n".join(items)
      • str that looks like [ ... ] → literal_eval → list → join
      • otherwise                   → str(value)
    """
    value = dict_obj.get(key, default)

    # 1) None / pandas NA
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default

    # 2) real list or tuple
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return "\n".join(str(item) for item in value)

    # 3) a string that *looks* like a Python list
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
                # fallback: strip [ ], split on commas, strip quotes
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

    # 4) scalar fallback
    return str(value)



# Sidebar navigation
with st.sidebar:
    st.markdown("## COMPETITIVE INTELLIGENCE")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Navigation menu
    pages = ['Overview', 'Programs', 'Companies', 'Clinical Trials', 'Compare', 'Cluster Analysis']
    selected_page = st.radio("Navigation", pages, key='nav_radio')
    st.session_state.current_page = selected_page
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption("Competitive Intelligence Viz v2.0")

plotly_layout = dict(
    plot_bgcolor='#FFFFFF',            # white plotting area
    paper_bgcolor='#FFFFFF',           # white page background
    font=dict(
        color='#2A3F5F',               # dark slate blue text
        family='Arial, sans-serif',    # clean, professional font
        size=12
    ),
    xaxis=dict(
        gridcolor='#E5E5E5',           # very light gray grid
        zerolinecolor='#E5E5E5',       # match grid for zero line
        showgrid=True,
        zeroline=True,
        linecolor='#CCCCCC',           # light axis line
        tickcolor='#2A3F5F'
    ),
    yaxis=dict(
        gridcolor='#E5E5E5',
        zerolinecolor='#E5E5E5',
        showgrid=True,
        zeroline=True,
        linecolor='#CCCCCC',
        tickcolor='#2A3F5F'
    ),
    colorway=[
        '#4C78A8',  # muted blue
        '#54A24B',  # medium green
        '#E45756',  # soft red
        '#79C36A',  # light green
        '#72B7B2',  # teal
        '#EECA3B',  # gold
        '#9B5DE5',  # purple
        '#7080A0'   # slate gray
    ],
    margin=dict(l=60, r=40, t=50, b=50),  # adjust padding if needed
    hoverlabel=dict(
        bgcolor='rgba(255,255,255,0.9)',
        font_color='#2A3F5F'
    )
)


# Main content area
if st.session_state.current_page == 'Overview':
    st.title("Overview")

    # File upload
    st.markdown("### Data Import")
    uploaded_file = st.file_uploader(
        "Select JSON dataset",
        type=['json', 'txt'],
        help="Upload competitive intelligence dataset in JSON format"
    )

    if uploaded_file is not None:
        st.session_state.data = load_data(uploaded_file)
        st.success("Data loaded successfully")

    if not st.session_state.data:
        st.info("Please upload a dataset to begin analysis")
        st.stop()

    # --- Base DataFrames ---
    data = st.session_state.data
    programs_df   = pd.DataFrame(data.get('programs', []))
    companies_df  = pd.DataFrame(data.get('companies', []))
    trials_df     = pd.DataFrame(data.get('trials', []))
    partnerships  = pd.DataFrame(data.get('partnerships', []))

    if programs_df.empty:
        st.warning("No program data found.")
        st.stop()

    # --- Filters (shown before plots) ---
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns(4)

    # Safely get unique options
    def uniq(series):
        if series.name not in programs_df.columns:
            return []
        return sorted([x for x in series.dropna().unique() if str(x).strip() != ""])

    with f1:
        sel_indications = st.multiselect(
            "Indication",
            options=uniq(programs_df['indication_group']) if 'indication_group' in programs_df else [],
            default=[],
            help="Leave empty to include all"
        )
    with f2:
        sel_targets = st.multiselect(
            "Target Family",
            options=uniq(programs_df['target_family_final']) if 'target_family_final' in programs_df else [],
            default=[],
        )
    with f3:
        sel_class = st.multiselect(
            "Classification",
            options=uniq(programs_df['program_classification_final']) if 'program_classification_final' in programs_df else [],
            default=[],
        )
    with f4:
        sel_modality = st.multiselect(
            "Modality",
            options=uniq(programs_df['modality_final']) if 'modality_final' in programs_df else [],
            default=[],
        )

    # --- Apply filters ---
    fdf = programs_df.copy()

    if sel_indications:
        fdf = fdf[fdf['indication_group'].isin(sel_indications)]
    if sel_targets:
        fdf = fdf[fdf['target_family_final'].isin(sel_targets)]
    if sel_class:
        fdf = fdf[fdf['program_classification_final'].isin(sel_class)]
    if sel_modality:
        fdf = fdf[fdf['modality_final'].isin(sel_modality)]

    # --- Derived sets for metrics ---
    filtered_company_names = set(fdf['company_name'].dropna().unique())
    filtered_program_ids   = set(fdf['program_id'].dropna().unique()) if 'program_id' in fdf else set()

    trials_f = pd.DataFrame()
    if not trials_df.empty and 'program_id' in trials_df.columns and filtered_program_ids:
        trials_f = trials_df[trials_df['program_id'].isin(filtered_program_ids)]

    partnerships_f = pd.DataFrame()
    if not partnerships.empty:
        # Count partnerships where at least one company is in the filtered set
        col_candidates = [c for c in partnerships.columns if 'company' in c.lower() or 'partner' in c.lower()]
        if col_candidates:
            mask = False
            for c in col_candidates:
                mask = mask | partnerships[c].isin(filtered_company_names)
            partnerships_f = partnerships[mask]

    # --- Key Metrics (filtered) ---
    st.markdown("### Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Companies", len(filtered_company_names))
    with m2:
        st.metric("Active Programs", len(fdf))
    with m3:
        st.metric("Clinical Trials", len(trials_f))
    with m4:
        st.metric("Partnerships", len(partnerships_f))

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Overview Plots (filtered) ---
    # Therapeutic Area Distribution
    st.markdown("### Therapeutic Area Distribution")
    if 'indication_group' in fdf.columns and not fdf.empty:
        ind_counts = fdf['indication_group'].value_counts().head(10)
        if not ind_counts.empty:
            fig_ind = px.bar(
                x=ind_counts.values,
                y=ind_counts.index,
                orientation='h',
                labels={'x': 'Number of Programs', 'y': 'Indication'},
                title="Leading Therapeutic Areas (filtered)"
            )
            fig_ind.update_layout(**plotly_layout, height=400)
            st.plotly_chart(fig_ind, use_container_width=True)
        else:
            st.info("No indications match the current filters.")
    else:
        st.info("No indication data available.")

    # Two-up: Stage + Modality (both computed from filtered programs)
    c1, c2 = st.columns(2)

    with c1:
        stage_col = 'development_stage_final' if 'development_stage_final' in fdf.columns else None
        if stage_col and not fdf.empty:
            stage_counts = fdf[stage_col].value_counts()
            if not stage_counts.empty:
                fig_stage = px.pie(
                    values=stage_counts.values,
                    names=stage_counts.index,
                    title="Development Stage Distribution (filtered)",
                    hole=0.4
                )
                fig_stage.update_layout(**plotly_layout)
                fig_stage.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_stage, use_container_width=True)
            else:
                st.info("No stages match the current filters.")
        else:
            st.info("No stage data available.")

    with c2:
        mod_col = 'modality_final' if 'modality_final' in fdf.columns else None
        if mod_col and not fdf.empty:
            mod_counts = fdf[mod_col].value_counts().head(8)
            if not mod_counts.empty:
                fig_mod = px.bar(
                    x=mod_counts.values,
                    y=mod_counts.index,
                    orientation='h',
                    title="Technology Platforms (filtered)"
                )
                fig_mod.update_layout(**plotly_layout)
                st.plotly_chart(fig_mod, use_container_width=True)
            else:
                st.info("No modalities match the current filters.")
        else:
            st.info("No modality data available.")




elif st.session_state.current_page == 'Programs' and st.session_state.data:
    data = st.session_state.data
    programs_df = pd.DataFrame(data.get('programs', []))
    companies_df = pd.DataFrame(data.get('companies', []))
    
    st.title("Drug Development Programs")
    
    # Filters
    cols = st.columns(6)

    def _opts(df, col):
        if col in df.columns:
            return ['All'] + sorted([x for x in df[col].dropna().unique() if str(x).strip() != ""])
        return ['All']

    with cols[0]:
        indications = _opts(programs_df, 'indication_group')
        selected_indication = st.selectbox("Indication", indications)

    with cols[1]:
        targets = _opts(programs_df, 'target_family_final')
        selected_target = st.selectbox("Target Family", targets)

    with cols[2]:
        classifications = _opts(programs_df, 'program_classification_final')
        selected_classification = st.selectbox("Classification", classifications)

    with cols[3]:
        modalities = _opts(programs_df, 'modality_final')
        selected_modality = st.selectbox("Modality", modalities)

    with cols[4]:
        deliveries = _opts(programs_df, 'platform_delivery_final')
        selected_delivery = st.selectbox("Delivery", deliveries)

    with cols[5]:
        stages = _opts(programs_df, 'development_stage_final')
        selected_stage = st.selectbox("Development Stage", stages)

    # Apply filters
    filtered_df = programs_df.copy()
    if selected_indication != 'All':
        filtered_df = filtered_df[filtered_df['indication_group'] == selected_indication]
    if selected_target != 'All':
        filtered_df = filtered_df[filtered_df['target_family_final'] == selected_target]
    if selected_classification != 'All':
        filtered_df = filtered_df[filtered_df['program_classification_final'] == selected_classification]
    if selected_modality != 'All':
        filtered_df = filtered_df[filtered_df['modality_final'] == selected_modality]
    if selected_delivery != 'All':
        filtered_df = filtered_df[filtered_df['platform_delivery_final'] == selected_delivery]
    if selected_stage != 'All':
        filtered_df = filtered_df[filtered_df['development_stage_final'] == selected_stage]

    
    st.markdown(f"### Results: {len(filtered_df)} programs")
    
    # Display programs as expandable cards
    for idx, program in filtered_df.iterrows():
        company = companies_df[companies_df['company_id'] == program['company_id']].iloc[0] if not companies_df[companies_df['company_id'] == program['company_id']].empty else {}
        
        with st.expander(f"{program['program_name'].upper()} - {program['company_name'].title()} | {program['development_stage_final']}"):
            
            # Company Info Section
            st.markdown("#### Company Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**Type:** {safe_get(company, 'public_private')}")
            with col2:
                st.markdown(f"**Country:** {safe_get(company, 'country_normalized')}")
            with col3:
                st.markdown(f"**Size:** {safe_get(company, 'size_category')}")
            with col4:
                st.markdown(f"**Founded:** {safe_get(company, 'founding_year', 'N/A')}")
            
            st.markdown("---")
            
            # Program Overview
            st.markdown("#### Program Details")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Program:** {program['program_name']}")
                st.markdown(f"**Classification:** {program['program_classification_final']}")
                st.markdown(f"**Target:** {program['target_primary']}")
            with col2:
                st.markdown(f"**Indication:** {program['indication_primary']}")
                st.markdown(f"**Delivery:** {program['platform_delivery_final']}")
                st.markdown(f"**Stage:** {program['development_stage_final']}")
            
            # Scientific Details
            st.markdown("#### Scientific Rationale")
            st.markdown(f"**Biological Rationale:** {safe_get(program, 'biological_rationale_final')}")
            st.markdown(f"**Mechanism of Action:** {safe_get(program, 'mechanism_of_action_detailed_final')}")
            
            # Clinical Trials
            trials = [t for t in data.get('trials', []) if t.get('program_id') == program['program_id']]
            if trials:
                st.markdown("#### Clinical Development")
                for trial in trials:
                    st.markdown(f"- **{trial_links(trial.get('trial_id'))}**: {safe_get(trial, 'phase')} – {safe_get(trial, 'status')}")
            

            st.markdown(f"**Milestones:** {safe_get(program, 'timeline_milestones')}")

                     
            # Additional Information
            st.markdown("#### Additional Information")
            st.markdown(f"**Research Notes:** {safe_get(program, 'research_notes')}")
            st.markdown(f"**Key Publications:** {safe_get(program, 'key_scientific_paper')}")
            st.markdown(f"**Data Quality Index:** {safe_get(program, 'data_quality_index')}")
            

            red_flags = safe_get(program, 'red_flags')
            if red_flags != 'N/A':
                st.warning(f"**Risk Factors:** {red_flags}")


elif st.session_state.current_page == 'Companies' and st.session_state.data:
    data = st.session_state.data
    companies_df = pd.DataFrame(data.get('companies', []))
    programs_df  = pd.DataFrame(data.get('programs',  []))
    trials_df    = pd.DataFrame(data.get('trials',    []))

    st.title("Companies")

    if companies_df.empty:
        st.info("No company data available.")
    else:
        # --- derived metrics ---
        prog_counts = programs_df['company_name'].value_counts() if not programs_df.empty else pd.Series(dtype=int)
        companies_df['active_programs'] = companies_df['company_name'].map(prog_counts).fillna(0).astype(int)

        if 'total_funding_numeric' in companies_df.columns:
            companies_df['total_funding_numeric'] = pd.to_numeric(companies_df['total_funding_numeric'], errors='coerce')

        # --- filters ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            name_query = st.text_input("Search name", "")
        with c2:
            types = ['All'] + sorted([x for x in companies_df['public_private'].dropna().unique()])
            sel_type = st.selectbox("Type", types)
        with c3:
            countries = ['All'] + sorted([x for x in companies_df['country_normalized'].dropna().unique()])
            sel_country = st.selectbox("Country", countries)
        with c4:
            sizes = ['All'] + sorted([x for x in companies_df['size_category'].dropna().unique()])
            sel_size = st.selectbox("Size", sizes)

        sort_by = st.selectbox("Sort by", ["Name (A→Z)", "Active programs (↓)", "Total funding (↓)"])

        # --- apply filters ---
        fdf = companies_df.copy()
        if name_query.strip():
            nq = name_query.strip().lower()
            fdf = fdf[fdf['company_name'].fillna("").str.lower().str.contains(nq)]
        if sel_type != 'All':
            fdf = fdf[fdf['public_private'] == sel_type]
        if sel_country != 'All':
            fdf = fdf[fdf['country_normalized'] == sel_country]
        if sel_size != 'All':
            fdf = fdf[fdf['size_category'] == sel_size]

        # --- sort ---
        if sort_by == "Name (A→Z)":
            fdf = fdf.sort_values(by='company_name', ascending=True, na_position='last')
        elif sort_by == "Active programs (↓)":
            fdf = fdf.sort_values(by='active_programs', ascending=False, na_position='last')
        else:
            if 'total_funding_numeric' in fdf.columns:
                fdf = fdf.sort_values(by='total_funding_numeric', ascending=False, na_position='last')

        st.markdown(f"### Results: {len(fdf)} companies")

        # --- list companies as expandable cards ---
        for _, comp in fdf.iterrows():
            name = comp.get('company_name', 'Unknown')
            headline = f"{str(name).title()} — {safe_get(comp, 'public_private')} | {safe_get(comp, 'country_normalized')} | {int(comp.get('active_programs', 0))} programs"

            with st.expander(headline):
                # Company Information
                st.markdown("#### Company Information")
                a, b, c, d = st.columns(4)
                with a:
                    st.markdown(f"**Type:** {safe_get(comp, 'public_private')}")
                with b:
                    st.markdown(f"**Country:** {safe_get(comp, 'country_normalized')}")
                with c:
                    st.markdown(f"**Size:** {safe_get(comp, 'size_category')}")
                with d:
                    st.markdown(f"**Founded:** {safe_get(comp, 'founding_year', 'N/A')}")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    tf = comp.get('total_funding_numeric', np.nan)
                    st.metric("Total Funding", format_currency(tf if pd.notna(tf) else 0))
                with m2:
                    st.metric("Active Programs", int(comp.get('active_programs', 0)))
                with m3:
                    st.metric("HQ City", safe_get(comp, 'city_normalized'))
                with m4:
                    # Smaller font for Primary Focus
                    st.markdown(
                        f"<div style='font-size: 0.85rem; line-height: 1.2em;'><b>Primary Focus:</b><br>{safe_get(comp, 'company_focus')}</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("---")

                # Programs
                st.markdown("#### Programs")
                comp_programs = programs_df[programs_df['company_name'] == name] if not programs_df.empty else pd.DataFrame()
                if comp_programs.empty:
                    st.info("No programs found for this company.")
                else:
                    for _, program in comp_programs.iterrows():
                        with st.expander(f"{program['program_name'].upper()} | {program['development_stage_final']}"):
                            cA, cB = st.columns(2)
                            with cA:
                                st.markdown(f"**Program:** {program['program_name']}")
                                st.markdown(f"**Classification:** {program['program_classification_final']}")
                                st.markdown(f"**Target:** {program['target_primary']}")
                            with cB:
                                st.markdown(f"**Indication:** {program['indication_primary']}")
                                st.markdown(f"**Delivery:** {program['platform_delivery_final']}")
                                st.markdown(f"**Stage:** {program['development_stage_final']}")

                            st.markdown("**Biological Rationale:** " + safe_get(program, 'biological_rationale_final'))
                            st.markdown("**Mechanism of Action:** " + safe_get(program, 'mechanism_of_action_detailed_final'))

                            prog_trials = trials_df[trials_df['program_id'] == program['program_id']] if not trials_df.empty else pd.DataFrame()
                            if not prog_trials.empty:
                                st.markdown("**Clinical Development:**")
                                for _, trial in prog_trials.iterrows():
                                    st.markdown(
                                        f"- **{trial_links(trial.get('trial_id'))}**: "
                                        f"{safe_get(trial, 'phase')} – {safe_get(trial, 'status')}"
                                    )

                            red_flags = safe_get(program, 'red_flags')
                            if red_flags != 'N/A':
                                st.warning(f"**Risk Factors:** {red_flags}")

                # Company-level trials
                st.markdown("#### Clinical Trials (Company Level)")
                comp_trials = trials_df[trials_df['company_name'] == name] if not trials_df.empty else pd.DataFrame()
                if comp_trials.empty:
                    st.info("No company-level trials found.")
                else:
                    for _, trial in comp_trials.iterrows():
                        st.markdown(f"- **{trial_links(trial.get('trial_id'))}**: {safe_get(trial, 'phase')} – {safe_get(trial, 'status')}")

                # Extra notes
                st.markdown("#### Additional Information")
                st.markdown(f"**Leadership:** {safe_get(comp, 'leadership')}")
                st.markdown(f"**Website:** {safe_get(comp, 'website')}")
                st.markdown(f"**Research Notes:** {safe_get(comp, 'research_notes')}")

elif st.session_state.current_page == 'Clinical Trials' and st.session_state.data:
    data = st.session_state.data
    trials_df = pd.DataFrame(data.get('trials', []))
    
    st.title("Clinical Trial Analysis")
    
    if not trials_df.empty:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            indications = ['All'] + list(trials_df['indication'].dropna().unique())
            selected_indication = st.selectbox("Indication", indications)
        
        with col2:
            phases = ['All'] + list(trials_df['phase'].dropna().unique())
            selected_phase = st.selectbox("Phase", phases)
        
        with col3:
            statuses = ['All'] + list(trials_df['status'].dropna().unique())
            selected_status = st.selectbox("Status", statuses)
        
        with col4:
            companies = ['All'] + list(trials_df['company_name'].dropna().unique())
            selected_company = st.selectbox("Sponsor", companies)
        
        # Apply filters
        filtered_trials = trials_df.copy()
        if selected_indication != 'All':
            filtered_trials = filtered_trials[filtered_trials['indication'] == selected_indication]
        if selected_phase != 'All':
            filtered_trials = filtered_trials[filtered_trials['phase'] == selected_phase]
        if selected_status != 'All':
            filtered_trials = filtered_trials[filtered_trials['status'] == selected_status]
        if selected_company != 'All':
            filtered_trials = filtered_trials[filtered_trials['company_name'] == selected_company]
        
        st.markdown(f"### Results: {len(filtered_trials)} trials")
        
        # Display trials
        for idx, trial in filtered_trials.iterrows():
            with st.expander(f"{safe_get(trial, 'trial_id')} - {trial['company_name'].title() if pd.notna(trial['company_name']) else 'Unknown'}"):
                # Dedicated link line (expander titles don't render links reliably)
                st.markdown(f"**ClinicalTrials.gov:** {trial_links(trial.get('trial_id'))}")

                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Program:** {safe_get(trial, 'program_name')}")
                    st.markdown(f"**Phase:** {safe_get(trial, 'phase')}")
                    st.markdown(f"**Status:** {safe_get(trial, 'status')}")
                    st.markdown(f"**Indication:** {safe_get(trial, 'indication')}")
                
                with col2:
                    st.markdown(f"**Sponsor:** {safe_get(trial, 'sponsor')}")
                    st.markdown(f"**Target Enrollment:** {safe_get(trial, 'enrollment_target')}")
                    st.markdown(f"**Countries:** {safe_get(trial, 'countries_normalized')}")
                    st.markdown(f"**Title:** {safe_get(trial, 'trial_title')}")

elif st.session_state.current_page == 'Compare' and st.session_state.data:
    data = st.session_state.data
    programs_df = pd.DataFrame(data.get('programs', []))
    companies_df = pd.DataFrame(data.get('companies', []))
    trials_df   = pd.DataFrame(data.get('trials',   []))

    st.title("Comparative Analysis")

    tab1, tab2 = st.tabs(["Portfolio Distribution", "Company Benchmarking"])

    # ————————————————
    # Tab 1: Cross‐Dimensional Matrix
    with tab1:
        st.markdown("### Portfolio Distribution")

        view = st.radio(
            "View",
            ["Company × Phase", "Cross-Dimensional Matrix (original)"],
            horizontal=True
        )

        # ---------- VIEW A: Company × Phase (Programs/Trials) ----------
        if view == "Company × Phase":
            src = st.radio("Analyze:", ["Programs"], horizontal=True)
            normalize = st.checkbox("Show as percentage within company", value=False)
            min_items = st.number_input("Min items per company (filter small players)", min_value=0, value=0, step=1)

            def _pick(df, name, prefs):
                for p in prefs:
                    if p in df.columns:
                        return p
                raise KeyError(f"Missing required column for {name}: tried {prefs}")

            # choose data source
            if src == "Trials" and 'trials_df' in locals() and isinstance(trials_df, pd.DataFrame) and not trials_df.empty:
                df = trials_df.copy()
                company_col = _pick(df, "company", ["company_name", "sponsor", "Sponsor"])
                phase_col   = _pick(df, "phase",   ["phase", "clinical_phase", "development_stage_final"])
                title_unit  = "trials"
            else:
                df = programs_df.copy()
                company_col = _pick(df, "company", ["company_name"])
                phase_col   = _pick(df, "phase",   ["development_stage_final"])
                title_unit  = "programs"

            # clean
            df = df[[company_col, phase_col]].copy()
            df[company_col] = df[company_col].astype(str).str.strip()
            df[phase_col]   = df[phase_col].astype(str).str.strip()

            phase_order = ["Discovery", "Preclinical", "Phase 1", "Phase 2", "Phase 3", "Filed", "Approved"]

            # aggregate
            agg = (
                df.groupby([company_col, phase_col], dropna=False)
                .size()
                .reset_index(name="count")
            )
            if agg.empty:
                st.warning("No data to visualize.")
                st.stop()

            totals = agg.groupby(company_col, as_index=False)['count'].sum().sort_values('count', ascending=False)
            if min_items > 0:
                keep = totals[totals['count'] >= min_items][company_col]
                agg  = agg[agg[company_col].isin(keep)]
                totals = totals[totals[company_col].isin(keep)]

            if totals.empty:
                st.info("All companies filtered out by the minimum threshold.")
                st.stop()

            top_default = min(10, len(totals))
            top_n = st.slider("Show top N companies by total volume", 3, min(50, len(totals)), top_default)
            top_companies = totals.head(top_n)[company_col].tolist()
            agg = agg[agg[company_col].isin(top_companies)]

            # order categories
            agg[company_col] = pd.Categorical(agg[company_col], categories=top_companies, ordered=True)
            phase_cats = [p for p in phase_order if p in agg[phase_col].unique().tolist()]
            if phase_cats:
                agg[phase_col] = pd.Categorical(
                    agg[phase_col],
                    categories=phase_cats + [p for p in agg[phase_col].unique() if p not in phase_cats],
                    ordered=True
                )

            # normalize option
            value_col, y_label = ("count", "Count")
            if normalize:
                agg["percent"] = agg["count"] / agg.groupby(company_col)["count"].transform("sum") * 100
                value_col, y_label = ("percent", "Share (%)")

            # stacked bar
            fig_bar = px.bar(
                agg,
                x=company_col, y=value_col, color=phase_col,
                labels={company_col: "Company", value_col: y_label, phase_col: "Phase"},
                title=f"{title_unit.capitalize()} by Phase per Company"
            )
            fig_bar.update_layout(**plotly_layout, barmode="stack", xaxis_title=None, legend_title="Phase", height=480)
            st.plotly_chart(fig_bar, use_container_width=True)

            # heatmap (use Option 1 margin fix)
            pivot = agg.pivot_table(index=company_col, columns=phase_col, values=value_col, fill_value=0)
            pivot = pivot.reindex(index=top_companies)
            if phase_cats:
                pivot = pivot.reindex(columns=phase_cats + [c for c in pivot.columns if c not in phase_cats])

            fig_heat = px.imshow(
                pivot,
                labels=dict(x="Phase", y="Company", color=y_label),
                title=f"{title_unit.capitalize()} by Phase per Company (Heatmap)",
                aspect="auto"
            )
            # remove conflicting key
            layout_no_margin = {k: v for k, v in plotly_layout.items() if k != 'margin'}
            fig_heat.update_layout(
                **layout_no_margin,
                height=520,
                margin=dict(l=60, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            with st.expander("Show aggregated table"):
                st.dataframe(
                    agg.sort_values([company_col, phase_col]).rename(
                        columns={company_col: "Company", phase_col: "Phase", value_col: y_label}
                    ),
                    use_container_width=True
                )

        # ---------- VIEW B: Your original Cross-Dimensional Matrix ----------
        else:
            st.markdown("### Cross-Dimensional Analysis")
            col1, col2 = st.columns(2)
            dimensions = [
                'indication_group', 'target_family_final',
                'program_classification_final', 'modality_final',
                'development_stage_final'
            ]
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
                fig.update_layout(**plotly_layout, height=600)
                st.plotly_chart(fig, use_container_width=True)


    # ————————————————
    # Tab 2: Direct Company Comparison (with full Program expanders + Trial lists)
    with tab2:
        st.markdown("### Direct Company Comparison")
        company_names = companies_df['company_name'].unique().tolist()

        col1, col2 = st.columns(2)
        with col1:
            company1 = st.selectbox("Company A", company_names, key='comp1')
        with col2:
            company2 = st.selectbox("Company B", company_names, key='comp2')

        if st.button("Compare"):
            # company summary and slices
            comp1_data     = companies_df.query("company_name == @company1").iloc[0]
            comp2_data     = companies_df.query("company_name == @company2").iloc[0]
            comp1_programs = programs_df.query("company_name == @company1")
            comp2_programs = programs_df.query("company_name == @company2")
            comp1_trials   = trials_df.query("company_name == @company1")
            comp2_trials   = trials_df.query("company_name == @company2")

            col1, col2 = st.columns(2)
            # — Company A —
            with col1:
                st.markdown(f"## {company1.title()}")
                st.markdown(f"**Type:** {safe_get(comp1_data, 'public_private')}")
                st.markdown(f"**Founded:** {safe_get(comp1_data, 'founding_year')}")
                st.markdown(f"**Total Funding:** {format_currency(comp1_data.get('total_funding_numeric', 0))}")
                st.markdown(f"**Active Programs:** {len(comp1_programs)}")
                st.markdown(f"**Focus Area:** {safe_get(comp1_data, 'company_focus')}")

                st.markdown("---")
                st.markdown("### Programs")
                for _, program in comp1_programs.iterrows():
                    with st.expander(f"{program['program_name'].upper()} | {program['development_stage_final']}"):
                        # Program Details
                        st.markdown("#### Program Details")
                        cA, cB = st.columns(2)
                        with cA:
                            st.markdown(f"**Program:** {program['program_name']}")
                            st.markdown(f"**Classification:** {program['program_classification_final']}")
                            st.markdown(f"**Target:** {program['target_primary']}")
                        with cB:
                            st.markdown(f"**Indication:** {program['indication_primary']}")
                            st.markdown(f"**Delivery:** {program['platform_delivery_final']}")
                            st.markdown(f"**Stage:** {program['development_stage_final']}")

                        st.markdown("#### Scientific Rationale")
                        st.markdown(f"**Biological Rationale:** {safe_get(program, 'biological_rationale_final')}")
                        st.markdown(f"**Mechanism of Action:** {safe_get(program, 'mechanism_of_action_detailed_final')}")

                        # Clinical Trials for this program
                        prog_trials = trials_df[trials_df['program_id'] == program['program_id']]
                        if not prog_trials.empty:
                            st.markdown("#### Clinical Development")
                            for _, trial in prog_trials.iterrows():
                                st.markdown(
                                    f"- **{trial_links(trial.get('trial_id'))}**: "
                                    f"{safe_get(trial, 'phase')} – {safe_get(trial, 'status')}"
                                )
                        st.markdown(f"**Milestones:** {safe_get(program, 'timeline_milestones')}")

                        st.markdown("#### Additional Information")
                        st.markdown(f"**Key Publications:** {safe_get(program, 'key_scientific_paper')}")
                        st.markdown(f"**Data Quality Index:** {safe_get(program, 'data_quality_index')}")
                        st.markdown(f"**Research Notes:** {safe_get(program, 'research_notes')}")

                        red_flags = safe_get(program, 'red_flags')
                        if red_flags != 'N/A':
                            st.warning(f"**Risk Factors:** {red_flags}")

                st.markdown("### Clinical Trials")
                if not comp1_trials.empty:
                    for _, trial in comp1_trials.iterrows():
                        st.markdown(
                            f"- **{trial_links(trial.get('trial_id'))}** | "
                            f"{safe_get(trial, 'phase')} • {safe_get(trial, 'status')} • "
                            f"{safe_get(trial, 'indication')} • "
                            f"Enroll: {safe_get(trial, 'enrollment_target')}"
                        )
                else:
                    st.info("No trials found")

            # — Company B —
            with col2:
                st.markdown(f"## {company2.title()}")
                st.markdown(f"**Type:** {safe_get(comp2_data, 'public_private')}")
                st.markdown(f"**Founded:** {safe_get(comp2_data, 'founding_year')}")
                st.markdown(f"**Total Funding:** {format_currency(comp2_data.get('total_funding_numeric', 0))}")
                st.markdown(f"**Active Programs:** {len(comp2_programs)}")
                st.markdown(f"**Focus Area:** {safe_get(comp2_data, 'company_focus')}")

                st.markdown("---")
                st.markdown("### Programs")
                for _, program in comp2_programs.iterrows():
                    with st.expander(f"{program['program_name'].upper()} | {program['development_stage_final']}"):
                        # repeat the same detail block as above
                        st.markdown("#### Program Details")
                        cA, cB = st.columns(2)
                        with cA:
                            st.markdown(f"**Program:** {program['program_name']}")
                            st.markdown(f"**Classification:** {program['program_classification_final']}")
                            st.markdown(f"**Target:** {program['target_primary']}")
                        with cB:
                            st.markdown(f"**Indication:** {program['indication_primary']}")
                            st.markdown(f"**Delivery:** {program['platform_delivery_final']}")
                            st.markdown(f"**Stage:** {program['development_stage_final']}")

                        st.markdown("#### Scientific Rationale")
                        st.markdown(f"**Biological Rationale:** {safe_get(program, 'biological_rationale_final')}")
                        st.markdown(f"**Mechanism of Action:** {safe_get(program, 'mechanism_of_action_detailed_final')}")

                        prog_trials = trials_df[trials_df['program_id'] == program['program_id']]
                        if not prog_trials.empty:
                            st.markdown("#### Clinical Development")
                            for _, trial in prog_trials.iterrows():
                                st.markdown(
                                    f"- **{trial_links(trial.get('trial_id'))}**: "
                                    f"{safe_get(trial, 'phase')} – {safe_get(trial, 'status')}"
                                )
                        
                        st.markdown(f"**Milestones:** {safe_get(program, 'timeline_milestones')}")

                        st.markdown("#### Additional Information")
                        st.markdown(f"**Key Publications:** {safe_get(program, 'key_scientific_paper')}")
                        st.markdown(f"**Data Quality Index:** {safe_get(program, 'data_quality_index')}")
                        st.markdown(f"**Research Notes:** {safe_get(program, 'research_notes')}")

                        red_flags = safe_get(program, 'red_flags')
                        if red_flags != 'N/A':
                            st.warning(f"**Risk Factors:** {red_flags}")

                st.markdown("### Clinical Trials")
                if not comp2_trials.empty:
                    for _, trial in comp2_trials.iterrows():
                        st.markdown(
                            f"- **{trial_links(trial.get('trial_id'))}** | "
                            f"{safe_get(trial, 'phase')} • {safe_get(trial, 'status')} • "
                            f"{safe_get(trial, 'indication')} • "
                            f"Enroll: {safe_get(trial, 'enrollment_target')}"
                        )
                else:
                    st.info("No trials found")








elif st.session_state.current_page == 'Cluster Analysis' and st.session_state.data:
    data = st.session_state.data
    programs_df = pd.DataFrame(data.get('programs', []))
    companies_df = pd.DataFrame(data.get('companies', []))
    
    st.title("Competitive Landscape Network")
    
    if programs_df.empty:
        st.warning("No program data available")
    else:
        # Network configuration
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            primary_dimension = st.selectbox(
                "Analysis dimension",
                ["Indication Group", "Target Family"]
            )
        
        with col2:
            if primary_dimension == "Indication Group":
                if 'indication_group' in programs_df.columns:
                    segment_counts = programs_df['indication_group'].value_counts()
                    segment_options = [f"{seg} ({count})" for seg, count in segment_counts.items() 
                                     if str(seg) != 'nan' and count >= 3]
                else:
                    segment_counts = programs_df['indication_group'].value_counts()
                    segment_options = [f"{seg} ({count})" for seg, count in segment_counts.items() 
                                     if str(seg) != 'nan' and count >= 3]
                
                if segment_options:
                    selected_segment = st.selectbox("Select indication", segment_options)
                    selected_segment = selected_segment.split(' (')[0] if selected_segment else None
                else:
                    st.warning("Insufficient data for network analysis")
                    selected_segment = None
                    
            else:
                if 'target_family_final' in programs_df.columns:
                    segment_counts = programs_df['target_family_final'].value_counts()
                    segment_options = [f"{seg} ({count})" for seg, count in segment_counts.items() 
                                     if str(seg) != 'nan' and count >= 3]
                else:
                    segment_counts = programs_df['target_primary'].value_counts()
                    segment_options = [f"{seg} ({count})" for seg, count in segment_counts.items() 
                                     if str(seg) != 'nan' and count >= 3]
                
                if segment_options:
                    selected_segment = st.selectbox("Select target", segment_options)
                    selected_segment = selected_segment.split(' (')[0] if selected_segment else None
                else:
                    st.warning("Insufficient data for network analysis")
                    selected_segment = None
        
        with col3:
            network_view = st.selectbox(
                "View type",
                ["Modality", "Platform", "Combined"]
            )
        
        if selected_segment:
            # Filter programs
            if primary_dimension == "Indication Group":
                if 'indication_group' in programs_df.columns:
                    segment_programs = programs_df[programs_df['indication_group'] == selected_segment]
                else:
                    segment_programs = programs_df[programs_df['indication_group'] == selected_segment]
            else:
                if 'target_family_final' in programs_df.columns:
                    segment_programs = programs_df[programs_df['target_family_final'] == selected_segment]
                else:
                    segment_programs = programs_df[programs_df['target_primary'] == selected_segment]
            
            # Build network
            G = nx.Graph()
            company_info = {}
            approach_nodes = set()
            company_nodes = set()
            
            # Node colors for dark theme with better contrast
            node_colors = {
                'Small molecule': plotly_layout['colorway'][0],   # '#4C78A8'
                'Antibody':        plotly_layout['colorway'][1],   # '#54A24B'
                'Cell therapy':    plotly_layout['colorway'][2],   # '#E45756'
                'Gene therapy':    plotly_layout['colorway'][3],   # '#79C36A'
                'RNA':             plotly_layout['colorway'][4],   # '#72B7B2'
                'Protein':         plotly_layout['colorway'][5],   # '#EECA3B'
                'Other':           plotly_layout['colorway'][6],   # '#9B5DE5'
            }

            if network_view == "Modality":
                for _, program in segment_programs.iterrows():
                    company = program['company_name']
                    modality = program['modality_final']
                    
                    if pd.notna(company) and pd.notna(modality):
                        if company not in company_nodes:
                            G.add_node(company, node_type='company', bipartite=0)
                            company_nodes.add(company)
                            
                            company_progs = segment_programs[segment_programs['company_name'] == company]
                            company_info[company] = {
                                'program_count': len(company_progs),
                                'modalities': company_progs['modality_final'].unique().tolist()
                            }
                        
                        if modality not in approach_nodes:
                            G.add_node(modality, node_type='modality', bipartite=1)
                            approach_nodes.add(modality)
                        
                        if G.has_edge(company, modality):
                            G[company][modality]['weight'] += 1
                        else:
                            G.add_edge(company, modality, weight=1)
            
            elif network_view == "Platform":
                for _, program in segment_programs.iterrows():
                    company = program['company_name']
                    platform = program['platform_delivery_final']
                    
                    if pd.notna(company) and pd.notna(platform):
                        if company not in company_nodes:
                            G.add_node(company, node_type='company', bipartite=0)
                            company_nodes.add(company)
                            
                            company_progs = segment_programs[segment_programs['company_name'] == company]
                            company_info[company] = {
                                'program_count': len(company_progs),
                                'platforms': company_progs['platform_delivery_final'].unique().tolist()
                            }
                        
                        if platform not in approach_nodes:
                            G.add_node(platform, node_type='platform', bipartite=1)
                            approach_nodes.add(platform)
                        
                        if G.has_edge(company, platform):
                            G[company][platform]['weight'] += 1
                        else:
                            G.add_edge(company, platform, weight=1)
            
            else:  # Combined
                for _, program in segment_programs.iterrows():
                    company = program['company_name']
                    modality = program['modality_final']
                    platform = program['platform_delivery_final']
                    
                    if pd.notna(company) and pd.notna(modality) and pd.notna(platform):
                        approach = f"{modality} / {platform}"
                        
                        if company not in company_nodes:
                            G.add_node(company, node_type='company', bipartite=0)
                            company_nodes.add(company)
                            
                            company_progs = segment_programs[segment_programs['company_name'] == company]
                            company_info[company] = {
                                'program_count': len(company_progs)
                            }
                        
                        if approach not in approach_nodes:
                            G.add_node(approach, node_type='approach', bipartite=1)
                            approach_nodes.add(approach)
                        
                        if G.has_edge(company, approach):
                            G[company][approach]['weight'] += 1
                        else:
                            G.add_edge(company, approach, weight=1)
            
            if len(G.nodes()) > 0:
                # Layout
                pos = nx.spring_layout(G, k=1.0, iterations=50, seed=42)
                
                # Create edge traces
                edge_traces = []
                for edge in G.edges(data=True):
                    if edge[0] in pos and edge[1] in pos:
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        weight = edge[2].get('weight', 1)
                        
                        edge_trace = go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode='lines',
                            line=dict(width=min(weight * 1.5, 6), color='#CCCCCC'),
                            opacity=0.7,
                            hoverinfo='none',
                            showlegend=False
                        )
                        edge_traces.append(edge_trace)
                
                # Company nodes
                company_x = []
                company_y = []
                company_text = []
                company_sizes = []
                
                for node in company_nodes:
                    if node in pos:
                        company_x.append(pos[node][0])
                        company_y.append(pos[node][1])
                        company_text.append(node[:20])
                        info = company_info.get(node, {})
                        company_sizes.append(20 + info.get('program_count', 1) * 4)
                
                company_trace = go.Scatter(
                    x=company_x,
                    y=company_y,
                    mode='markers+text',
                    text=company_text,
                    textposition="top center",
                    textfont=dict(size=10, color=plotly_layout['font']['color']),
                    marker=dict(
                        size=company_sizes,
                        color=plotly_layout['colorway'][0],
                        line=dict(width=2, color='#999999')
                    ),
                    name='Companies',
                    hoverinfo='text',
                    hovertext=company_text
                )
                
                # Approach nodes
                approach_x = []
                approach_y = []
                approach_text = []
                approach_colors = []
                approach_sizes = []
                
                for node in approach_nodes:
                    if node in pos:
                        approach_x.append(pos[node][0])
                        approach_y.append(pos[node][1])
                        approach_text.append(node[:20])
                        
                        connected = list(G.neighbors(node))
                        approach_sizes.append(25 + len(connected) * 3)
                        
                        if network_view == "Modality":
                            approach_colors.append(node_colors.get(node, '#6b7280'))
                        else:
                            approach_colors.append('#fb923c')
                
                approach_trace = go.Scatter(
                    x=approach_x,
                    y=approach_y,
                    mode='markers+text',
                    text=approach_text,
                    textposition="bottom center",
                    textfont=dict(size=11, color=plotly_layout['font']['color']),
                    marker=dict(
                        size=approach_sizes,
                        color=[ node_colors.get(n, plotly_layout['colorway'][7]) for n in approach_nodes ],
                        symbol='diamond',
                        line=dict(width=2, color='#FFFFFF')
                    ),
                    name='Approaches',
                    hoverinfo='text',
                    hovertext=approach_text
                )
                            
                # Create figure
                fig = go.Figure(
                    data=edge_traces + [company_trace, approach_trace],
                    layout=go.Layout(
                        title=f'{selected_segment}: {network_view} Network',
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=20, r=20, t=60),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=700,
                        plot_bgcolor=plotly_layout['plot_bgcolor'],
                        paper_bgcolor=plotly_layout['paper_bgcolor'],
                        font=plotly_layout['font'],
                        dragmode='pan'
                    )
                )
                            
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Companies", len(company_nodes))
                
                with col2:
                    st.metric("Unique Approaches", len(approach_nodes))
                
                with col3:
                    st.metric("Total Programs", len(segment_programs))
            else:
                st.warning("Insufficient data for network visualization")
        else:
            st.info("Select an indication or target to analyze the competitive landscape")


else:
    if st.session_state.current_page != 'Overview':
        st.warning("Please upload a dataset from the Overview page")
