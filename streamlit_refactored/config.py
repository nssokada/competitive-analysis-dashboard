# ======================== config.py ========================
"""Configuration settings for the application"""

import streamlit as st

class PageConfig:
    """Page configuration settings"""
    
    @staticmethod
    def setup():
        st.set_page_config(
            page_title="Competitive Intelligence Platform",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        st.markdown("""
            <style>
            div[data-testid="metric-container"] { 
                background-color: #252830;
                border: 1px solid #3a3f4b;
                border-radius: 10px;
                padding: 10px;
            }
            .streamlit-expanderHeader { 
                background: #1f2229; 
                color: #e6e6e6; 
                border-radius: 6px; 
            }
            .streamlit-expanderContent { 
                background: #232730; 
                border: 1px solid #343a46; 
                border-radius: 0 0 6px 6px;
            }
            .section-divider {
                margin: 20px 0;
                border-bottom: 1px solid #3a3f4b;
            }
            </style>
        """, unsafe_allow_html=True)

class PlotlyTheme:
    """Plotly theme configuration"""
    
    LAYOUT = dict(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(
            color='#2A3F5F',
            family='Arial, sans-serif',
            size=12
        ),
        xaxis=dict(
            gridcolor='#E5E5E5',
            zerolinecolor='#E5E5E5',
            showgrid=True,
            zeroline=True,
            linecolor='#CCCCCC',
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
            '#4C78A8', '#54A24B', '#E45756', '#79C36A',
            '#72B7B2', '#EECA3B', '#9B5DE5', '#7080A0'
        ],
        margin=dict(l=60, r=40, t=50, b=50),
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.9)',
            font_color='#2A3F5F'
        )
    )