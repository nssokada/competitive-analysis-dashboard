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
    pages = ['Overview', 'Programs', 'Clinical Trials', 'Compare', 'Cluster Analysis']
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
    
    # File upload section
    st.markdown("### Data Import")
    uploaded_file = st.file_uploader(
        "Select JSON dataset", 
        type=['json', 'txt'],
        help="Upload competitive intelligence dataset in JSON format"
    )
    
    if uploaded_file is not None:
        st.session_state.data = load_data(uploaded_file)
        st.success("Data loaded successfully")
    
    if st.session_state.data:
        data = st.session_state.data
        
        # Display high-level statistics
        st.markdown("### Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Companies", len(data.get('companies', [])))
        
        with col2:
            st.metric("Active Programs", len(data.get('programs', [])))
        
        with col3:
            st.metric("Clinical Trials", len(data.get('trials', [])))
        
        with col4:
            st.metric("Partnerships", len(data.get('partnerships', [])))
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Programs by indication
        st.markdown("### Therapeutic Area Distribution")
        programs_df = pd.DataFrame(data.get('programs', []))
        if not programs_df.empty:
            indication_counts = programs_df['indication_group'].value_counts().head(10)
            fig_ind = px.bar(
                x=indication_counts.values, 
                y=indication_counts.index,
                orientation='h',
                labels={'x': 'Number of Programs', 'y': 'Indication'},
                title="Leading Therapeutic Areas"
            )
            fig_ind.update_layout(**plotly_layout, height=400)
            st.plotly_chart(fig_ind, use_container_width=True)
        
        # Programs by development stage
        col1, col2 = st.columns(2)
        
        with col1:
            if 'summary_stats' in data and 'programs_by_stage' in data['summary_stats']:
                stage_data = data['summary_stats']['programs_by_stage']
                fig_stage = px.pie(
                    values=list(stage_data.values()),
                    names=list(stage_data.keys()),
                    title="Development Stage Distribution",
                    hole=0.4
                )
                fig_stage.update_layout(**plotly_layout)
                fig_stage.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_stage, use_container_width=True)
        
        with col2:
            if 'summary_stats' in data and 'programs_by_modality' in data['summary_stats']:
                modality_data = data['summary_stats']['programs_by_modality']
                top_modalities = dict(sorted(modality_data.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:8])
                fig_mod = px.bar(
                    x=list(top_modalities.values()),
                    y=list(top_modalities.keys()),
                    orientation='h',
                    title="Technology Platforms"
                )
                fig_mod.update_layout(**plotly_layout)
                st.plotly_chart(fig_mod, use_container_width=True)
    
    else:
        st.info("Please upload a dataset file to begin analysis")

elif st.session_state.current_page == 'Programs' and st.session_state.data:
    data = st.session_state.data
    programs_df = pd.DataFrame(data.get('programs', []))
    companies_df = pd.DataFrame(data.get('companies', []))
    
    st.title("Drug Development Programs")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        indications = ['All'] + list(programs_df['indication_group'].unique())
        selected_indication = st.selectbox("Indication", indications)
    
    with col2:
        targets = ['All'] + list(programs_df['target_family_final'].unique())
        selected_target = st.selectbox("Target Family", targets)
    
    with col3:
        classifications = ['All'] + list(programs_df['program_classification_final'].unique())
        selected_classification = st.selectbox("Classification", classifications)
    
    with col4:
        modalities = ['All'] + list(programs_df['modality_final'].unique())
        selected_modality = st.selectbox("Modality", modalities)
    
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
                    st.markdown(f"- **{safe_get(trial, 'trial_id')}**: {safe_get(trial, 'phase')} - {safe_get(trial, 'status')}")
            

            st.markdown(f"**Milestones:** {safe_get(program, 'timeline_milestones')}")

                     
            # Additional Information
            st.markdown("#### Additional Information")
            st.markdown(f"**Research Notes:** {safe_get(program, 'research_notes')}")
            st.markdown(f"**Key Publications:** {safe_get(program, 'key_scientific_paper')}")
            st.markdown(f"**Data Quality Index:** {safe_get(program, 'data_quality_index')}")
            

            red_flags = safe_get(program, 'red_flags')
            if red_flags != 'N/A':
                st.warning(f"**Risk Factors:** {red_flags}")

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
        st.markdown("### Cross-Dimensional Analysis")
        col1, col2 = st.columns(2)
        dimensions = [
            'indication_group', 'target_family_final',
            'program_classification_final', 'modality_final',
            'development_stage_final'
        ]
        with col1:
            x_dim = st.selectbox("X-axis", dimensions)
        with col2:
            y_dim = st.selectbox("Y-axis", dimensions)

        if st.button("Generate Analysis"):
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
                                    f"- **{safe_get(trial, 'trial_id')}**: "
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
                            f"- **{safe_get(trial, 'trial_id')}** | "
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
                                    f"- **{safe_get(trial, 'trial_id')}**: "
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
                            f"- **{safe_get(trial, 'trial_id')}** | "
                            f"{safe_get(trial, 'phase')} • {safe_get(trial, 'status')} • "
                            f"{safe_get(trial, 'indication')} • "
                            f"Enroll: {safe_get(trial, 'enrollment_target')}"
                        )
                else:
                    st.info("No trials found")


elif st.session_state.current_page == 'Network Analysis' and st.session_state.data:
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