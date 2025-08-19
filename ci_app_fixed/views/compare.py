
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from core import state as S
from core.utils import safe_get, format_currency
from theme import PLOTLY_LAYOUT
from core.io import load_company_infos_blob
from components.company_extras import render_company_extras_for_company
from components.publications import render_publications_for_program

def render():
    data = st.session_state[S.DATA]
    programs_df = pd.DataFrame(data.get('programs', []))
    companies_df = pd.DataFrame(data.get('companies', []))
    trials_df_norm = st.session_state.get(S.TRIALS_DF_NORM)

    if S.COMPANY_INFOS not in st.session_state or not st.session_state[S.COMPANY_INFOS]:
        st.session_state[S.COMPANY_INFOS] = load_company_infos_blob(data or {})

    pubs_df = st.session_state.get(S.PUBLICATIONS_DF, pd.DataFrame())
    company_infos = st.session_state.get(S.COMPANY_INFOS, {})


    st.title("Comparative Analysis")
    tab1, tab2 = st.tabs(["Portfolio Distribution", "Company Benchmarking"])

    with tab1:
        st.markdown("### Portfolio Distribution")
        view = st.radio("View", ["Company × Phase", "Cross-Dimensional Matrix"], horizontal=True)
        if view == "Company × Phase":
            min_items = st.number_input("Min items per company (filter small players)", min_value=0, value=0, step=1)

            df = programs_df.copy()
            if df.empty:
                st.warning("No data to visualize."); return
            company_col = "company_name"
            phase_col = "development_stage_final"
            df = df[[company_col, phase_col]].copy()
            df[company_col] = df[company_col].astype(str).str.strip()
            df[phase_col]   = df[phase_col].astype(str).str.strip()
            phase_order = ["Discovery","Preclinical","Phase 1","Phase 2","Phase 3","Filed","Approved"]

            agg = df.groupby([company_col, phase_col], dropna=False).size().reset_index(name="count")
            totals = agg.groupby(company_col, as_index=False)['count'].sum().sort_values('count', ascending=False)
            if min_items > 0:
                keep = totals[totals['count'] >= min_items][company_col]
                agg  = agg[agg[company_col].isin(keep)]
                totals = totals[totals[company_col].isin(keep)]
            if totals.empty:
                st.info("All companies filtered out by the minimum threshold."); return

            top_default = min(10, len(totals))
            top_n = st.slider("Show top N companies by total volume", 3, min(50, len(totals)), top_default)
            top_companies = totals.head(top_n)[company_col].tolist()
            agg = agg[agg[company_col].isin(top_companies)]
            agg[company_col] = pd.Categorical(agg[company_col], categories=top_companies, ordered=True)
            phase_cats = [p for p in phase_order if p in agg[phase_col].unique().tolist()]
            if phase_cats:
                agg[phase_col] = pd.Categorical(agg[phase_col], categories=phase_cats + [p for p in agg[phase_col].unique() if p not in phase_cats], ordered=True)

            fig_bar = px.bar(
                agg, x=company_col, y="count", color=phase_col,
                labels={company_col:"Company", "count":"Count", phase_col:"Phase"},
                title="Programs by Phase per Company"
            )
            fig_bar.update_layout(**PLOTLY_LAYOUT, barmode="stack", xaxis_title=None, legend_title="Phase", height=480)
            st.plotly_chart(fig_bar, use_container_width=True)

            pivot = agg.pivot_table(index=company_col, columns=phase_col, values="count", fill_value=0)
            pivot = pivot.reindex(index=top_companies)
            if phase_cats:
                pivot = pivot.reindex(columns=phase_cats + [c for c in pivot.columns if c not in phase_cats])
            fig_heat = px.imshow(pivot, labels=dict(x="Phase", y="Company", color="Count"), title="Programs by Phase per Company (Heatmap)", aspect="auto")
            fig_heat.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k!='margin'}, height=520, margin=dict(l=60,r=40,t=60,b=40))
            st.plotly_chart(fig_heat, use_container_width=True)

            with st.expander("Show aggregated table"):
                st.dataframe(agg.sort_values([company_col, phase_col]).rename(columns={company_col:"Company", phase_col:"Phase", "count":"Count"}), use_container_width=True)

        else:
            st.markdown("### Cross-Dimensional Analysis")
            dimensions = ['indication_group','target_family_final','program_classification_final','modality_final','development_stage_final']
            col1, col2 = st.columns(2)
            with col1: x_dim = st.selectbox("X-axis", dimensions, key="x_dim_matrix")
            with col2: y_dim = st.selectbox("Y-axis", dimensions, key="y_dim_matrix")
            if st.button("Generate Analysis", key="gen_matrix"):
                cross_tab = pd.crosstab(programs_df[y_dim], programs_df[x_dim])
                fig = px.imshow(cross_tab, labels=dict(x=x_dim, y=y_dim, color="Count"), title="Portfolio Distribution Matrix", aspect="auto", color_continuous_scale="Blues")
                fig.update_layout(**PLOTLY_LAYOUT, height=600)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Company Benchmarking")
        company_names = sorted(programs_df.get('company_name', pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        c0, c1 = st.columns([4,2])
        with c0:
            selected = st.multiselect("Select companies (up to 6)", options=company_names, default=company_names[:min(2, len(company_names))])
        with c1:
            cols_per_row = st.slider("Columns", 2, 4, 3, step=1)
        if not selected: st.stop()
        if len(selected) > 6:
            st.warning("Showing the first 6 selected companies.")
            selected = selected[:6]

        def _ci_eq(series, val):
            s = series.astype(str).str.strip().str.lower()
            return s == str(val).strip().lower()

        prog_cols = [c for c in ['program_name','development_stage_final','indication_primary','program_classification_final','modality_final','platform_delivery_final','target_primary'] if c in programs_df.columns]
        trial_cols = [c for c in ['trial_id','phase','status','indication','enrollment_target','program_name','nct_id'] if (trials_df_norm is not None and c in trials_df_norm.columns)]

        def company_card(name: str):
            c = companies_df[_ci_eq(companies_df['company_name'], name)]
            cdict = c.iloc[0].to_dict() if not c.empty else {}
            progs  = programs_df[_ci_eq(programs_df['company_name'], name)] if not programs_df.empty else pd.DataFrame()
            trials = trials_df_norm[_ci_eq(trials_df_norm['company_name'], name)] if (trials_df_norm is not None and not trials_df_norm.empty) else pd.DataFrame()

            title = str(name).title()
            if len(title) > 48: title = title[:45] + "…"
            st.markdown(f"## {title}")
            with st.expander("Key Metrics", expanded=False):
                st.markdown(f"**Type:** {safe_get(cdict, 'public_private')}")
                st.markdown(f"**Founded:** {safe_get(cdict, 'founding_year')}")
                st.markdown(f"**Total Funding:** {format_currency(cdict.get('total_funding_numeric'))}")
                st.markdown(f"**Active Programs:** {int(progs.shape[0])}")
            with st.expander("Company Details", expanded=False):
                st.markdown(f"**Country:** {safe_get(cdict, 'country_normalized')}")
                st.markdown(f"**HQ City:** {safe_get(cdict, 'city_normalized')}")
                st.markdown(f"**Website:** {safe_get(cdict, 'website')}")
                st.markdown(f"**Size:** {safe_get(cdict, 'size_category')}")
                st.markdown(f"**Primary Focus:** {safe_get(cdict, 'company_focus')}")
                st.markdown(f"**Leadership:** {safe_get(cdict, 'leadership')}")
            with st.expander("Programs", expanded=False):
                if progs.empty:
                    st.info("No programs found.")
                else:
                    view = progs[prog_cols] if prog_cols else progs
                    st.dataframe(view, use_container_width=True, hide_index=True)
            with st.expander("Clinical Trials", expanded=False):
                if trials.empty:
                    st.info("No trials found.")
                else:
                    view = trials.drop_duplicates(subset=['trial_key'])
                    if trial_cols: view = view[trial_cols].copy()
                    st.dataframe(view, use_container_width=True, hide_index=True)
            
            # --- Strategic relationships & funding (company_extras) ---
            with st.expander("Strategic Relationships & Funding", expanded=False):
                if not company_infos:
                    st.info("No company extras loaded.")
                else:
                    ok = render_company_extras_for_company(cdict, company_infos)
                    if not ok:
                        st.info("No funding, partnerships, M&A, or news found for this company.")

            # --- Publications (program-level) ---
            with st.expander("Publications", expanded=False):
                if progs is None or progs.empty:
                    st.info("No programs available for this company.")
                else:
                    # Let the user pick a program to keep this compact
                    prog_names = (
                        progs.get("program_name")
                        .fillna("Unnamed program")
                        .astype(str)
                        .tolist()
                    )
                    default_idx = 0
                    sel_prog = st.selectbox(
                        "Select a program to view publications",
                        prog_names,
                        index=default_idx,
                        key=f"pubs_select_{name}"
                    )
                    # Fetch the selected program row
                    sel_row = progs[progs["program_name"].astype(str) == sel_prog]
                    if sel_row.empty:
                        st.info("No matching program row found.")
                    else:
                        prog_row_dict = sel_row.iloc[0].to_dict()
                        ok = render_publications_for_program(prog_row_dict, pubs_df)
                        if not ok:
                            st.info("No publications found for this program.")

            with st.expander("Notes", expanded=False):
                st.markdown(f"**Research Notes:** {safe_get(cdict, 'research_notes')}")

        for start in range(0, len(selected), cols_per_row):
            row = selected[start:start+cols_per_row]
            cols = st.columns(len(row))
            for i, cname in enumerate(row):
                with cols[i]:
                    company_card(cname)
