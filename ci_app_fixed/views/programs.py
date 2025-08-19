
from __future__ import annotations
import streamlit as st
import pandas as pd
from core import state as S
from core.io import build_search_blob
from core.utils import safe_get
from components.publications import render_publications_for_program

def _opts(df, col):
    if col in df.columns:
        return ['All'] + sorted([x for x in df[col].dropna().unique() if str(x).strip() != ""])
    return ['All']

def render():
    data = st.session_state[S.DATA]
    programs_df = pd.DataFrame(data.get('programs', []))
    companies_df = pd.DataFrame(data.get('companies', []))
    trials_df_norm = st.session_state.get(S.TRIALS_DF_NORM)

    programs_df = build_search_blob(programs_df)

    st.title("Drug Development Programs")
    search_query = st.text_input(
        "Search programs…",
        value="",
        placeholder="Try: 'liposarcoma MDM2 Rain' or 'KRAS G12C Amgen'",
    )

    cols = st.columns(6)
    with cols[0]:
        selected_indication = st.selectbox("Indication", _opts(programs_df,'indication_group'))
    with cols[1]:
        selected_target = st.selectbox("Target Family", _opts(programs_df,'target_family_final'))
    with cols[2]:
        selected_classification = st.selectbox("Classification", _opts(programs_df,'program_classification_final'))
    with cols[3]:
        selected_modality = st.selectbox("Modality", _opts(programs_df,'modality_final'))
    with cols[4]:
        selected_delivery = st.selectbox("Delivery", _opts(programs_df,'platform_delivery_final'))
    with cols[5]:
        selected_stage = st.selectbox("Development Stage", _opts(programs_df,'development_stage_final'))

    filtered_df = programs_df.copy()
    if search_query.strip():
        terms = [t.lower() for t in search_query.split() if t.strip()]
        if terms:
            mask = pd.Series(True, index=filtered_df.index)
            blob = filtered_df['_search_blob']
            for t in terms:
                mask &= blob.str.contains(t, regex=False)
            filtered_df = filtered_df[mask]
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

    pubs_df = st.session_state.get(S.PUBLICATIONS_DF, pd.DataFrame())

    for _, program in filtered_df.iterrows():
        company_row = companies_df[companies_df['company_id'] == program['company_id']]
        company = company_row.iloc[0].to_dict() if not company_row.empty else {}

        with st.expander(f"{program['program_name'].upper()} - {program['company_name'].title()} | {program['development_stage_final']}"):
            st.markdown("#### Company Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.markdown(f"**Type:** {safe_get(company,'public_private')}")
            with col2: st.markdown(f"**Country:** {safe_get(company,'country_normalized')}")
            with col3: st.markdown(f"**Size:** {safe_get(company,'size_category')}")
            with col4: st.markdown(f"**Founded:** {safe_get(company,'founding_year','N/A')}")

            st.markdown("---")
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

            st.markdown("#### Scientific Rationale")
            st.markdown(f"**Biological Rationale:** {safe_get(program,'biological_rationale_final')}")
            st.markdown(f"**Mechanism of Action:** {safe_get(program,'mechanism_of_action_detailed_final')}")

            # Trials
            prog_trials = pd.DataFrame()
            if trials_df_norm is not None and not trials_df_norm.empty:
                prog_trials = trials_df_norm[trials_df_norm['program_id'] == program['program_id']]
            if not prog_trials.empty:
                st.markdown("#### Clinical Development")
                for _, tr in prog_trials.drop_duplicates(subset=['trial_key']).iterrows():
                    if pd.notna(tr['nct_id']) and tr['nct_id']:
                        nct_display = f"[{tr['nct_id']}](https://clinicaltrials.gov/study/{tr['nct_id']})"
                    else:
                        nct_display = "N/A"
                    st.markdown(f"- **{nct_display}**: {program.get('development_stage_final','')} – {tr.get('status_clean','')}")

            st.markdown(f"**Milestones:** {safe_get(program,'timeline_milestones')}")

            st.markdown("#### Additional Information")
            st.markdown(f"**Research Notes:** {safe_get(program,'research_notes')}")
            st.markdown(f"**Key Publications:** {safe_get(program,'key_scientific_paper')}")
            st.markdown(f"**Data Quality Index:** {safe_get(program,'data_quality_index')}")

            _ = render_publications_for_program(program, pubs_df)

            red_flags = safe_get(program,'red_flags')
            if red_flags != 'N/A':
                st.warning(f"**Risk Factors:** {red_flags}")
