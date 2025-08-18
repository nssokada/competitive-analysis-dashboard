# ======================== pages/cluster_analysis.py ========================
"""Cluster Analysis page"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from pages.base import BasePage
from config import PlotlyTheme

class ClusterAnalysisPage(BasePage):
    """Competitive Landscape Network Analysis page"""
    
    def _render_content(self):
        st.title("Competitive Landscape Network")
        
        programs_df = self.data_loader.programs_df
        
        if programs_df.empty:
            st.warning("No program data available")
            return
        
        # Configure analysis
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            primary_dimension = st.selectbox(
                "Analysis dimension",
                ["Indication Group", "Target Family"]
            )
        
        with col2:
            selected_segment = self._get_segment_selection(programs_df, primary_dimension)
        
        with col3:
            network_view = st.selectbox(
                "View type",
                ["Modality", "Platform", "Combined"]
            )
        
        if selected_segment:
            self._render_network(programs_df, primary_dimension, selected_segment, network_view)
        else:
            st.info("Select an indication or target to analyze the competitive landscape")
    
    def _get_segment_selection(self, programs_df, dimension):
        """Get segment selection based on dimension"""
        if dimension == "Indication Group":
            column = 'indication_group' if 'indication_group' in programs_df.columns else 'indication_group'
        else:
            column = 'target_family_final' if 'target_family_final' in programs_df.columns else 'target_primary'
        
        segment_counts = programs_df[column].value_counts()
        segment_options = [
            f"{seg} ({count})"
            for seg, count in segment_counts.items()
            if str(seg) != 'nan' and count >= 3
        ]
        
        if segment_options:
            selected = st.selectbox(
                f"Select {'indication' if dimension == 'Indication Group' else 'target'}",
                segment_options
            )
            return selected.split(' (')[0] if selected else None
        else:
            st.warning("Insufficient data for network analysis")
            return None
    
    def _render_network(self, programs_df, dimension, segment, view):
        """Render network visualization"""
        # Filter programs
        if dimension == "Indication Group":
            column = 'indication_group' if 'indication_group' in programs_df.columns else 'indication_group'
        else:
            column = 'target_family_final' if 'target_family_final' in programs_df.columns else 'target_primary'
        
        segment_programs = programs_df[programs_df[column] == segment]
        
        # Build network
        G = nx.Graph()
        company_info = {}
        approach_nodes = set()
        company_nodes = set()
        
        # Node colors
        node_colors = {
            'Small molecule': PlotlyTheme.LAYOUT['colorway'][0],
            'Antibody': PlotlyTheme.LAYOUT['colorway'][1],
            'Cell therapy': PlotlyTheme.LAYOUT['colorway'][2],
            'Gene therapy': PlotlyTheme.LAYOUT['colorway'][3],
            'RNA': PlotlyTheme.LAYOUT['colorway'][4],
            'Protein': PlotlyTheme.LAYOUT['colorway'][5],
            'Other': PlotlyTheme.LAYOUT['colorway'][6],
        }
        
        # Build graph based on view type
        self._build_network_graph(
            G, segment_programs, view,
            company_nodes, approach_nodes, company_info
        )
        
        if len(G.nodes()) == 0:
            st.warning("Insufficient data for network visualization")
            return
        
        # Create visualization
        fig = self._create_network_figure(
            G, company_nodes, approach_nodes,
            node_colors, segment, view
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Companies", len(company_nodes))
        with col2:
            st.metric("Unique Approaches", len(approach_nodes))
        with col3:
            st.metric("Total Programs", len(segment_programs))
    
    def _build_network_graph(self, G, programs, view, company_nodes, approach_nodes, company_info):
        """Build network graph based on view type"""
        if view == "Modality":
            node_field = 'modality_final'
            node_type = 'modality'
        elif view == "Platform":
            node_field = 'platform_delivery_final'
            node_type = 'platform'
        else:  # Combined
            node_field = None
            node_type = 'approach'
        
        for _, program in programs.iterrows():
            company = program['company_name']
            
            if view == "Combined":
                modality = program['modality_final']
                platform = program['platform_delivery_final']
                if pd.notna(company) and pd.notna(modality) and pd.notna(platform):
                    approach = f"{modality} / {platform}"
                else:
                    continue
            else:
                approach = program[node_field]
                if pd.isna(company) or pd.isna(approach):
                    continue
            
            # Add company node
            if company not in company_nodes:
                G.add_node(company, node_type='company', bipartite=0)
                company_nodes.add(company)
                company_progs = programs[programs['company_name'] == company]
                company_info[company] = {
                    'program_count': len(company_progs)
                }
            
            # Add approach node
            if approach not in approach_nodes:
                G.add_node(approach, node_type=node_type, bipartite=1)
                approach_nodes.add(approach)
            
            # Add or update edge
            if G.has_edge(company, approach):
                G[company][approach]['weight'] += 1
            else:
                G.add_edge(company, approach, weight=1)
    
    def _create_network_figure(self, G, company_nodes, approach_nodes, node_colors, segment, view):
        """Create plotly network figure"""
        # Generate layout
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
        company_x, company_y, company_text, company_sizes = [], [], [], []
        for node in company_nodes:
            if node in pos:
                company_x.append(pos[node][0])
                company_y.append(pos[node][1])
                company_text.append(node[:20])
                company_sizes.append(20 + G.degree(node) * 4)
        
        company_trace = go.Scatter(
            x=company_x,
            y=company_y,
            mode='markers+text',
            text=company_text,
            textposition="top center",
            textfont=dict(size=10, color=PlotlyTheme.LAYOUT['font']['color']),
            marker=dict(
                size=company_sizes,
                color=PlotlyTheme.LAYOUT['colorway'][0],
                line=dict(width=2, color='#999999')
            ),
            name='Companies',
            hoverinfo='text',
            hovertext=company_text
        )
        
        # Approach nodes
        approach_x, approach_y, approach_text = [], [], []
        for node in approach_nodes:
            if node in pos:
                approach_x.append(pos[node][0])
                approach_y.append(pos[node][1])
                approach_text.append(node[:20])
        
        approach_trace = go.Scatter(
            x=approach_x,
            y=approach_y,
            mode='markers+text',
            text=approach_text,
            textposition="bottom center",
            textfont=dict(size=11, color=PlotlyTheme.LAYOUT['font']['color']),
            marker=dict(
                size=[25 + len(list(G.neighbors(n))) * 3 for n in approach_nodes],
                color=[node_colors.get(n, PlotlyTheme.LAYOUT['colorway'][7]) for n in approach_nodes],
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
                title=f'{segment}: {view} Network',
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,
                plot_bgcolor=PlotlyTheme.LAYOUT['plot_bgcolor'],
                paper_bgcolor=PlotlyTheme.LAYOUT['paper_bgcolor'],
                font=PlotlyTheme.LAYOUT['font'],
                dragmode='pan'
            )
        )
        
        return fig