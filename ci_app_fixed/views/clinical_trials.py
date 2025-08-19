
from __future__ import annotations
import streamlit as st
import pandas as pd
from core import state as S
from core.utils import safe_get, trial_links

def render():
    trials_df_norm = st.session_state.get(S.TRIALS_DF_NORM)

    st.title("Clinical Trial Analysis")
    if trials_df_norm is None or trials_df_norm.empty:
        st.info("No trial data available.")
        return

    trials_df_view = trials_df_norm.copy()

    phase_options = ['All']
    if 'phase_list' in trials_df_view.columns:
        phase_atomic = sorted({p for lst in trials_df_view['phase_list'] for p in (lst or [])})
        phase_options += phase_atomic

    status_options = ['All']
    if 'status_list' in trials_df_view.columns:
        status_atomic = sorted({s for lst in trials_df_view['status_list'] for s in (lst or [])})
        status_options += status_atomic

    indication_options = ['All'] + sorted(trials_df_view['indication'].dropna().unique()) if 'indication' in trials_df_view.columns else ['All']
    company_options   = ['All'] + sorted(trials_df_view['company_name'].dropna().unique()) if 'company_name' in trials_df_view.columns else ['All']

    col1, col2, col3, col4 = st.columns(4)
    with col1: selected_indication = st.selectbox("Indication", indication_options)
    with col2: selected_phase = st.selectbox("Phase", phase_options)
    with col3: selected_status = st.selectbox("Status", status_options)
    with col4: selected_company = st.selectbox("Sponsor", company_options)

    filtered_trials = trials_df_view.copy()
    if selected_indication != 'All' and 'indication' in filtered_trials.columns:
        filtered_trials = filtered_trials[filtered_trials['indication'] == selected_indication]
    if selected_phase != 'All' and 'phase_list' in filtered_trials.columns:
        filtered_trials = filtered_trials[filtered_trials['phase_list'].apply(lambda L: selected_phase in (L or []))]
    if selected_status != 'All' and 'status_list' in filtered_trials.columns:
        filtered_trials = filtered_trials[filtered_trials['status_list'].apply(lambda L: selected_status in (L or []))]
    if selected_company != 'All' and 'company_name' in filtered_trials.columns:
        filtered_trials = filtered_trials[filtered_trials['company_name'] == selected_company]

    dedup = filtered_trials.drop_duplicates(subset=['trial_key'])
    st.markdown(f"### Results: {len(dedup)} trials")

    for _, trial in dedup.iterrows():
        title_left = (trial['nct_id'] if pd.notna(trial['nct_id']) and trial['nct_id'] else trial['trial_label'])
        sponsor_disp = trial['company_name'].title() if pd.notna(trial.get('company_name')) else 'Unknown'

        with st.expander(f"{title_left} - {sponsor_disp}"):
            link_source = trial.get('nct_id') or trial.get('trial_id')
            st.markdown(f"**ClinicalTrials.gov:** {trial_links(link_source)}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Program:** {safe_get(trial, 'program_name')}")
                st.markdown(f"**Status:** {trial.get('status_clean','')}")
                st.markdown(f"**Indication:** {safe_get(trial, 'indication')}")
            with c2:
                st.markdown(f"**Sponsor:** {safe_get(trial, 'sponsor')}")
                st.markdown(f"**Countries:** {trial.get('countries_clean','')}")
                st.markdown(f"**Title:** {trial.get('trial_title_clean','')}")
