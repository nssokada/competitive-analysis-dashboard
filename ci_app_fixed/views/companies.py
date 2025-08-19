
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from core import state as S
from core.io import load_company_infos_blob
from core.utils import safe_get, format_currency
from components.company_extras import render_company_extras_for_company
from components.publications import render_publications_for_program

def render():
    data = st.session_state[S.DATA]
    if S.COMPANY_INFOS not in st.session_state or not st.session_state[S.COMPANY_INFOS]:
        st.session_state[S.COMPANY_INFOS] = load_company_infos_blob(data or {})

    companies_df = pd.DataFrame(data.get('companies', []))
    programs_df  = pd.DataFrame(data.get('programs',  []))
    pubs_df      = st.session_state.get(S.PUBLICATIONS_DF, pd.DataFrame())
    trials_df_norm = st.session_state.get(S.TRIALS_DF_NORM)

    st.title("Companies")
    st.caption(f"Loaded extras for {len(st.session_state.get(S.COMPANY_INFOS, {}))} companies")

    if companies_df.empty:
        st.info("No company data available.")
        return

    prog_counts = programs_df['company_name'].value_counts() if not programs_df.empty else pd.Series(dtype=int)
    companies_df['active_programs'] = companies_df['company_name'].map(prog_counts).fillna(0).astype(int)

    if 'total_funding_numeric' in companies_df.columns:
        companies_df['total_funding_numeric'] = pd.to_numeric(companies_df['total_funding_numeric'], errors='coerce')

    c1, c2, c3, c4 = st.columns(4)
    with c1: name_query = st.text_input("Search name", "")
    with c2: sel_type = st.selectbox("Type", ['All'] + sorted([x for x in companies_df['public_private'].dropna().unique()]))
    with c3: sel_country = st.selectbox("Country", ['All'] + sorted([x for x in companies_df['country_normalized'].dropna().unique()]))
    with c4: sel_size = st.selectbox("Size", ['All'] + sorted([x for x in companies_df['size_category'].dropna().unique()]))
    sort_by = st.selectbox("Sort by", ["Name (A→Z)", "Active programs (↓)", "Total funding (↓)"])

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

    if sort_by == "Name (A→Z)":
        fdf = fdf.sort_values(by='company_name', ascending=True, na_position='last')
    elif sort_by == "Active programs (↓)":
        fdf = fdf.sort_values(by='active_programs', ascending=False, na_position='last')
    else:
        if 'total_funding_numeric' in fdf.columns:
            fdf = fdf.sort_values(by='total_funding_numeric', ascending=False, na_position='last')

    st.markdown(f"### Results: {len(fdf)} companies")

    for _, comp in fdf.iterrows():
        name = comp.get('company_name', 'Unknown')
        headline = f"{str(name).title()} — {safe_get(comp, 'public_private')} | {safe_get(comp, 'country_normalized')} | {int(comp.get('active_programs', 0))} programs"
        with st.expander(headline):
            st.markdown("#### Company Information")
            a, b, c, d = st.columns(4)
            with a: st.markdown(f"**Type:** {safe_get(comp,'public_private')}")
            with b: st.markdown(f"**Country:** {safe_get(comp,'country_normalized')}")
            with c: st.markdown(f"**Size:** {safe_get(comp,'size_category')}")
            with d: st.markdown(f"**Founded:** {safe_get(comp,'founding_year','N/A')}")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                tf = comp.get('total_funding_numeric', np.nan)
                st.metric("Total Funding", format_currency(tf if pd.notna(tf) else 0))
            with m2: st.metric("Active Programs", int(comp.get('active_programs', 0)))
            with m3: st.metric("HQ City", safe_get(comp, 'city_normalized'))
            with m4: st.markdown(f"<div style='font-size: 0.85rem; line-height: 1.2em;'><b>Primary Focus:</b><br>{safe_get(comp, 'company_focus')}</div>", unsafe_allow_html=True)

            st.markdown("---")
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

                        _ = render_publications_for_program(program, pubs_df)

                        prog_trials = pd.DataFrame()
                        if trials_df_norm is not None and not trials_df_norm.empty:
                            prog_trials = trials_df_norm[trials_df_norm['program_id'] == program['program_id']]
                        if not prog_trials.empty:
                            st.markdown("**Clinical Development:**")
                            for _, tr in prog_trials.drop_duplicates(subset=['trial_key']).iterrows():
                                nct_display = f"[{tr['nct_id']}](https://clinicaltrials.gov/study/{tr['nct_id']})" if pd.notna(tr['nct_id']) and tr['nct_id'] else "N/A"
                                st.markdown(f"- **{nct_display}**: {tr.get('phase_clean','')} – {tr.get('status_clean','')}")

                        red_flags = safe_get(program, 'red_flags')
                        if red_flags != 'N/A':
                            st.warning(f"**Risk Factors:** {red_flags}")

            st.markdown("#### Clinical Trials (Company Level)")
            comp_trials = trials_df_norm[trials_df_norm['company_name'] == name] if (trials_df_norm is not None and not trials_df_norm.empty) else pd.DataFrame()
            if comp_trials.empty:
                st.info("No company-level trials found.")
            else:
                for _, trial in comp_trials.drop_duplicates(subset=['trial_key']).iterrows():
                    nct_display = f"[{trial['nct_id']}](https://clinicaltrials.gov/study/{trial['nct_id']})" if pd.notna(trial['nct_id']) and trial['nct_id'] else "N/A"
                    st.markdown(f"- **{nct_display}**: {trial.get('phase_clean','')} – {trial.get('status_clean','')}")

            st.markdown("#### Additional Information")
            st.markdown(f"**Leadership:** {safe_get(comp, 'leadership')}")
            st.markdown(f"**Website:** {safe_get(comp, 'website')}")
            st.markdown(f"**Research Notes:** {safe_get(comp, 'research_notes')}")

            extras_shown = render_company_extras_for_company(comp.to_dict(), st.session_state.get(S.COMPANY_INFOS, {}))
            if not extras_shown and bool(comp.get("has_company_info", False)):
                st.info("Additional company intelligence is available in your dataset but not yet loaded. Ensure a top-level 'company_infos' is present in the uploaded JSON or provide a local extras directory.")
