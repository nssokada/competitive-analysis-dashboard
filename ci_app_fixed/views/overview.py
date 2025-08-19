
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from core import state as S
from core.io import load_data, normalize_trials, load_company_infos_blob
from core.utils import format_currency
from theme import PLOTLY_LAYOUT

def _uniq(df: pd.DataFrame, col: str):
    if col not in df.columns: return []
    return sorted([x for x in df[col].dropna().unique() if str(x).strip() != ""])

def render():
    st.title("Overview")

    st.markdown("### Data Import")
    uploaded_file = st.file_uploader(
        "Select JSON dataset",
        type=['json','txt'],
        help="Upload competitive intelligence dataset in JSON format"
    )
    if uploaded_file is not None:
        st.session_state[S.DATA] = load_data(uploaded_file.read())
        st.success("Data loaded successfully")
        data = st.session_state[S.DATA] or {}
        st.session_state[S.PUBLICATIONS_DF] = pd.DataFrame(data.get('publications', []))
        trials_df_raw = pd.DataFrame(data.get('trials', []))
        st.session_state[S.TRIALS_DF_NORM] = normalize_trials(trials_df_raw)

    if not st.session_state.get(S.DATA):
        st.info("Please upload a dataset to begin analysis")
        st.stop()

    data = st.session_state[S.DATA]
    programs_df   = pd.DataFrame(data.get('programs', []))
    companies_df  = pd.DataFrame(data.get('companies', []))
    trials_df_raw = pd.DataFrame(data.get('trials', []))
    trials_df_norm = st.session_state.get(S.TRIALS_DF_NORM)
    if trials_df_norm is None or trials_df_norm.empty:
        trials_df_norm = normalize_trials(trials_df_raw)

    partnerships  = pd.DataFrame(data.get('partnerships', []))
    st.session_state[S.COMPANY_INFOS] = load_company_infos_blob(data)

    if programs_df.empty:
        st.warning("No program data found."); st.stop()

    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sel_indications = st.multiselect("Indication", options=_uniq(programs_df,'indication_group'), default=[])
    with f2:
        sel_targets = st.multiselect("Target Family", options=_uniq(programs_df,'target_family_final'), default=[])
    with f3:
        sel_class = st.multiselect("Classification", options=_uniq(programs_df,'program_classification_final'), default=[])
    with f4:
        sel_modality = st.multiselect("Modality", options=_uniq(programs_df,'modality_final'), default=[])

    fdf = programs_df.copy()
    if sel_indications: fdf = fdf[fdf['indication_group'].isin(sel_indications)]
    if sel_targets:     fdf = fdf[fdf['target_family_final'].isin(sel_targets)]
    if sel_class:       fdf = fdf[fdf['program_classification_final'].isin(sel_class)]
    if sel_modality:    fdf = fdf[fdf['modality_final'].isin(sel_modality)]

    filtered_company_names = set(fdf['company_name'].dropna().unique())
    filtered_program_ids   = set(fdf.get('program_id', pd.Series(dtype=str)).dropna().unique())

    trials_f = trials_df_norm[trials_df_norm['program_id'].isin(filtered_program_ids)] if not trials_df_norm.empty and filtered_program_ids else pd.DataFrame()

    partnerships_f = pd.DataFrame()
    if not partnerships.empty:
        col_candidates = [c for c in partnerships.columns if 'company' in c.lower() or 'partner' in c.lower()]
        if col_candidates:
            mask = False
            for c in col_candidates:
                mask = mask | partnerships[c].isin(filtered_company_names)
            partnerships_f = partnerships[mask]

    st.markdown("### Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Total Companies", len(filtered_company_names))
    with m2: st.metric("Active Programs", len(fdf))
    with m3: st.metric("Clinical Trials", len(trials_f.drop_duplicates(subset=['trial_key'])) if not trials_f.empty else 0)
    with m4: st.metric("Partnerships", len(partnerships_f))

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Therapeutic Area Distribution")
    if 'indication_group' in fdf.columns and not fdf.empty:
        ind_counts = fdf['indication_group'].value_counts().head(10)
        if not ind_counts.empty:
            fig_ind = px.bar(
                x=ind_counts.values, y=ind_counts.index, orientation='h',
                labels={'x':'Number of Programs','y':'Indication'},
                title="Leading Therapeutic Areas (filtered)"
            )
            fig_ind.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig_ind, use_container_width=True)
        else:
            st.info("No indications match the current filters.")
    else:
        st.info("No indication data available.")

    c1, c2 = st.columns(2)
    with c1:
        stage_col = 'development_stage_final' if 'development_stage_final' in fdf.columns else None
        if stage_col and not fdf.empty:
            stage_counts = fdf[stage_col].value_counts()
            if not stage_counts.empty:
                fig_stage = px.pie(values=stage_counts.values, names=stage_counts.index, title="Development Stage Distribution (filtered)", hole=0.4)
                fig_stage.update_layout(**PLOTLY_LAYOUT)
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
                fig_mod = px.bar(x=mod_counts.values, y=mod_counts.index, orientation='h', title="Technology Platforms (filtered)")
                fig_mod.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_mod, use_container_width=True)
            else:
                st.info("No modalities match the current filters.")
        else:
            st.info("No modality data available.")
