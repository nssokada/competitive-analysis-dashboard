# ======================== pages/programs.py ========================
"""Programs page"""

import streamlit as st
from pages.base import BasePage
from components import ProgramCard, FilterBar

class ProgramsPage(BasePage):
    """Drug Development Programs page"""
    
    def _render_content(self):
        st.title("Drug Development Programs")
        
        programs_df = self.data_loader.programs_df
        companies_df = self.data_loader.companies_df
        trials_df = self.data_loader.trials_df_norm
        
        # Search bar
        search_query = st.text_input(
            "Search programs…",
            value="",
            placeholder="Try: 'liposarcoma MDM2 Rain' or 'KRAS G12C Amgen'"
        )
        
        # Filters
        filter_config = [
            ('indication', 'indication_group', 'Indication'),
            ('target', 'target_family_final', 'Target Family'),
            ('classification', 'program_classification_final', 'Classification'),
            ('modality', 'modality_final', 'Modality'),
            ('delivery', 'platform_delivery_final', 'Delivery'),
            ('stage', 'development_stage_final', 'Development Stage')
        ]
        
        filters = FilterBar.render_selectbox(programs_df, filter_config)
        
        # Apply filters
        filtered_df = self._apply_filters(programs_df, search_query, filters)
        
        # Display results
        st.markdown(f"### Results: {len(filtered_df)} programs")
        
        for _, program in filtered_df.iterrows():
            # Get company info
            company_row = companies_df[companies_df['company_id'] == program['company_id']]
            company = company_row.iloc[0].to_dict() if not company_row.empty else None
            
            # Render program card
            ProgramCard.render(
                program,
                company=company,
                trials_df=trials_df,
                publications_df=self.data_loader.publications_df
            )
    
    def _apply_filters(self, df, search_query, filters):
        """Apply search and filters to programs dataframe"""
        filtered_df = df.copy()
        
        # Apply search
        if search_query.strip():
            terms = [t.lower() for t in search_query.split() if t.strip()]
            if terms:
                mask = True
                for term in terms:
                    mask &= filtered_df['_search_blob'].str.contains(term, regex=False)
                filtered_df = filtered_df[mask]
        
        # Apply filters
        for key, value in filters.items():
            if value != 'All':
                col_map = {
                    'indication': 'indication_group',
                    'target': 'target_family_final',
                    'classification': 'program_classification_final',
                    'modality': 'modality_final',
                    'delivery': 'platform_delivery_final',
                    'stage': 'development_stage_final'
                }
                if key in col_map:
                    filtered_df = filtered_df[filtered_df[col_map[key]] == value]
        
        return filtered_df
