"""
Competitive Intelligence Platform
A streamlit application for analyzing drug development programs, companies, and clinical trials.
"""

import streamlit as st
from config import PageConfig, PlotlyTheme
from data_loader import DataLoader
from pages import (
    OverviewPage,
    ProgramsPage,
    CompaniesPage,
    ClinicalTrialsPage,
    ComparePage,
    ClusterAnalysisPage
)
from components import Sidebar

# -------------------------------
# Page Configuration
# -------------------------------
PageConfig.setup()

# -------------------------------
# Initialize Session State
# -------------------------------
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Overview'

# -------------------------------
# Sidebar Navigation
# -------------------------------
sidebar = Sidebar()
selected_page = sidebar.render()
st.session_state.current_page = selected_page

# -------------------------------
# Page Routing
# -------------------------------
pages = {
    'Overview': OverviewPage(),
    'Programs': ProgramsPage(),
    'Companies': CompaniesPage(),
    'Clinical Trials': ClinicalTrialsPage(),
    'Compare': ComparePage(),
    'Cluster Analysis': ClusterAnalysisPage()
}

# Render selected page
if selected_page in pages:
    pages[selected_page].render()
else:
    st.error("Page not found")

