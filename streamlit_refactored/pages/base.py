# ======================== pages/base.py ========================
"""Base page class"""

import streamlit as st
from abc import ABC, abstractmethod

class BasePage(ABC):
    """Abstract base class for pages"""
    
    def __init__(self):
        self.data_loader = None
    
    def render(self):
        """Main render method"""
        # Get data loader from session state
        self.data_loader = st.session_state.get('data_loader')
        
        # Check if data is loaded
        if not self.data_loader or not self.data_loader.data:
            if self.__class__.__name__ != 'OverviewPage':
                st.warning("Please upload a dataset from the Overview page")
                return
        
        # Call page-specific render
        self._render_content()
    
    @abstractmethod
    def _render_content(self):
        """Page-specific content rendering"""
        pass