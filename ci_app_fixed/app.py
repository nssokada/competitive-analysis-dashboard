
from __future__ import annotations
import streamlit as st
from theme import inject, PLOTLY_LAYOUT
from core import state as S
from views import overview, programs, companies, clinical_trials, compare, cluster

# --- Page config ---
st.set_page_config(page_title="Competitive Intelligence Platform", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
inject()

# --- Ensure session keys exist ---
if S.DATA not in st.session_state: st.session_state[S.DATA] = None
if S.CURRENT_PAGE not in st.session_state: st.session_state[S.CURRENT_PAGE] = S.DEFAULT_PAGE
if S.TRIALS_DF_NORM not in st.session_state: 
    import pandas as pd
    st.session_state[S.TRIALS_DF_NORM] = pd.DataFrame()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## COMPETITIVE INTELLIGENCE")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    pages = ['Overview','Programs','Companies','Clinical Trials','Compare','Cluster Analysis']
    selected_page = st.radio("Navigation", pages, key='nav_radio')
    st.session_state[S.CURRENT_PAGE] = selected_page
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption("Competitive Intelligence Viz v2.0")

# --- Router ---
page = st.session_state[S.CURRENT_PAGE]
if page == 'Overview':
    overview.render()
else:
    if not st.session_state.get(S.DATA):
        st.warning("Please upload a dataset from the Overview page")
    else:
        if page == 'Programs':
            programs.render()
        elif page == 'Companies':
            companies.render()
        elif page == 'Clinical Trials':
            clinical_trials.render()
        elif page == 'Compare':
            compare.render()
        elif page == 'Cluster Analysis':
            cluster.render()
        else:
            overview.render()
