
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from core import state as S
from theme import PLOTLY_LAYOUT

def render():
    data = st.session_state[S.DATA]
    programs_df = pd.DataFrame(data.get('programs', []))

    st.title("Competitive Landscape Network")
    if programs_df.empty:
        st.warning("No program data available"); return

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        primary_dimension = st.selectbox("Analysis dimension", ["Indication Group", "Target Family"])
    with col2:
        if primary_dimension == "Indication Group":
            if 'indication_group' in programs_df.columns:
                segment_counts = programs_df['indication_group'].value_counts()
            else:
                segment_counts = programs_df['indication_group'].value_counts()
            segment_options = [f"{seg} ({count})" for seg, count in segment_counts.items() if str(seg) != 'nan' and count >= 3]
            selected_segment = st.selectbox("Select indication", segment_options) if segment_options else None
            selected_segment = selected_segment.split(' (')[0] if selected_segment else None
        else:
            if 'target_family_final' in programs_df.columns:
                segment_counts = programs_df['target_family_final'].value_counts()
            else:
                segment_counts = programs_df['target_primary'].value_counts()
            segment_options = [f"{seg} ({count})" for seg, count in segment_counts.items() if str(seg) != 'nan' and count >= 3]
            selected_segment = st.selectbox("Select target", segment_options) if segment_options else None
            selected_segment = selected_segment.split(' (')[0] if selected_segment else None
    with col3:
        network_view = st.selectbox("View type", ["Modality", "Platform", "Combined"])

    if not selected_segment:
        st.info("Select an indication or target to analyze the competitive landscape"); return

    if primary_dimension == "Indication Group":
        if 'indication_group' in programs_df.columns:
            segment_programs = programs_df[programs_df['indication_group'] == selected_segment]
        else:
            segment_programs = programs_df[programs_df['indication_group'] == selected_segment]
    else:
        if 'target_family_final' in programs_df.columns:
            segment_programs = programs_df[programs_df['target_family_final'] == selected_segment]
        else:
            segment_programs = programs_df[programs_df['target_primary'] == selected_segment]

    G = nx.Graph()
    company_info = {}
    approach_nodes, company_nodes = set(), set()
    node_colors = {
        'Small molecule': PLOTLY_LAYOUT['colorway'][0],
        'Antibody':        PLOTLY_LAYOUT['colorway'][1],
        'Cell therapy':    PLOTLY_LAYOUT['colorway'][2],
        'Gene therapy':    PLOTLY_LAYOUT['colorway'][3],
        'RNA':             PLOTLY_LAYOUT['colorway'][4],
        'Protein':         PLOTLY_LAYOUT['colorway'][5],
        'Other':           PLOTLY_LAYOUT['colorway'][6],
    }

    if network_view == "Modality":
        for _, program in segment_programs.iterrows():
            company = program['company_name']
            modality = program['modality_final']
            if pd.notna(company) and pd.notna(modality):
                if company not in company_nodes:
                    G.add_node(company, node_type='company', bipartite=0)
                    company_nodes.add(company)
                    company_progs = segment_programs[segment_programs['company_name'] == company]
                    company_info[company] = {'program_count': len(company_progs), 'modalities': company_progs['modality_final'].unique().tolist()}
                if modality not in approach_nodes:
                    G.add_node(modality, node_type='modality', bipartite=1)
                    approach_nodes.add(modality)
                G.add_edge(company, modality, weight=G.get_edge_data(company, modality, {}).get('weight', 0) + 1)

    elif network_view == "Platform":
        for _, program in segment_programs.iterrows():
            company = program['company_name']
            platform = program['platform_delivery_final']
            if pd.notna(company) and pd.notna(platform):
                if company not in company_nodes:
                    G.add_node(company, node_type='company', bipartite=0)
                    company_nodes.add(company)
                    company_progs = segment_programs[segment_programs['company_name'] == company]
                    company_info[company] = {'program_count': len(company_progs), 'platforms': company_progs['platform_delivery_final'].unique().tolist()}
                if platform not in approach_nodes:
                    G.add_node(platform, node_type='platform', bipartite=1)
                    approach_nodes.add(platform)
                G.add_edge(company, platform, weight=G.get_edge_data(company, platform, {}).get('weight', 0) + 1)

    else:
        for _, program in segment_programs.iterrows():
            company = program['company_name']
            modality = program['modality_final']
            platform = program['platform_delivery_final']
            if pd.notna(company) and pd.notna(modality) and pd.notna(platform):
                approach = f"{modality} / {platform}"
                if company not in company_nodes:
                    G.add_node(company, node_type='company', bipartite=0); company_nodes.add(company)
                    company_info[company] = {'program_count': len(segment_programs[segment_programs['company_name'] == company])}
                if approach not in approach_nodes:
                    G.add_node(approach, node_type='approach', bipartite=1); approach_nodes.add(approach)
                G.add_edge(company, approach, weight=G.get_edge_data(company, approach, {}).get('weight', 0) + 1)

    if len(G.nodes()) == 0:
        st.warning("Insufficient data for network visualization"); return

    pos = nx.spring_layout(G, k=1.0, iterations=50, seed=42)
    edge_traces = []
    for u, v, data in G.edges(data=True):
        if u in pos and v in pos:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            weight = data.get('weight', 1)
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines', line=dict(width=min(weight*1.5, 6), color='#CCCCCC'),
                opacity=0.7, hoverinfo='none', showlegend=False
            ))

    company_x, company_y, company_text, company_sizes = [], [], [], []
    for node in company_nodes:
        if node in pos:
            company_x.append(pos[node][0]); company_y.append(pos[node][1])
            company_text.append(node[:20])
            info = company_info.get(node, {})
            company_sizes.append(20 + info.get('program_count', 1) * 4)
    company_trace = go.Scatter(
        x=company_x, y=company_y, mode='markers+text', text=company_text,
        textposition="top center", textfont=dict(size=10, color=PLOTLY_LAYOUT['font']['color']),
        marker=dict(size=company_sizes, color=PLOTLY_LAYOUT['colorway'][0], line=dict(width=2, color='#999999')),
        name='Companies', hoverinfo='text', hovertext=company_text
    )

    approach_x, approach_y, approach_text = [], [], []
    for node in approach_nodes:
        if node in pos:
            approach_x.append(pos[node][0]); approach_y.append(pos[node][1])
            approach_text.append(node[:20])
    approach_trace = go.Scatter(
        x=approach_x, y=approach_y, mode='markers+text', text=approach_text,
        textposition="bottom center", textfont=dict(size=11, color=PLOTLY_LAYOUT['font']['color']),
        marker=dict(size=[25 + len(list(G.neighbors(n))) * 3 for n in approach_nodes],
                    color=[PLOTLY_LAYOUT['colorway'][7] for _ in approach_nodes],
                    symbol='diamond', line=dict(width=2, color='#FFFFFF')),
        name='Approaches', hoverinfo='text', hovertext=approach_text
    )

    fig = go.Figure(data=edge_traces + [company_trace, approach_trace],
                    layout=go.Layout(
                        title=f'{selected_segment}: {network_view} Network',
                        showlegend=True, hovermode='closest',
                        margin=dict(b=20,l=20,r=20,t=60),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=700,
                        plot_bgcolor=PLOTLY_LAYOUT['plot_bgcolor'],
                        paper_bgcolor=PLOTLY_LAYOUT['paper_bgcolor'],
                        font=PLOTLY_LAYOUT['font'],
                        dragmode='pan'
                    ))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Companies", len(company_nodes))
    with c2: st.metric("Unique Approaches", len(approach_nodes))
    with c3: st.metric("Total Programs", len(segment_programs))
