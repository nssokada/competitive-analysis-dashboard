
from __future__ import annotations
import pandas as pd
import streamlit as st
from typing import Optional
from core.utils import norm_str

_WORK_TYPE_LABEL = {
    "journal": "Journal articles",
    "conf_abstract": "Conference abstracts",
    "poster": "Posters",
    "preprint": "Preprints",
    "whitepaper": "Whitepapers / decks",
}
_WORK_TYPE_ORDER = ["journal", "conf_abstract", "poster", "preprint", "whitepaper"]

def _best_link(row: pd.Series) -> Optional[str]:
    url = str(row.get("url") or "").strip()
    if url:
        return url
    doi = str(row.get("doi") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    return None

def render_publications_for_program(program_row, pubs_df: pd.DataFrame) -> bool:
    if pubs_df is None or pubs_df.empty:
        return False

    prog_id = str(program_row.get("program_id"))
    prog_name = norm_str(program_row.get("program_name"))
    comp_name = norm_str(program_row.get("company_name"))

    m_id = pubs_df["program_id"].astype(str).eq(prog_id) if "program_id" in pubs_df.columns else pd.Series([False]*len(pubs_df))
    m_pn = pubs_df.get("program_name", pd.Series([""]*len(pubs_df))).astype(str).str.lower().eq(prog_name)
    m_cn = pubs_df.get("company_name", pd.Series([""]*len(pubs_df))).astype(str).str.lower().str.contains(comp_name)

    sel = pubs_df[m_id | (m_pn & m_cn)]
    if sel.empty: return False

    st.markdown("#### Publications")
    types_present = [wt for wt in _WORK_TYPE_ORDER if wt in sel["work_type"].dropna().str.lower().unique().tolist()]
    other_types = sorted(set(sel["work_type"].dropna().str.lower()) - set(types_present))
    groups = types_present + other_types

    chips = []
    for wt in groups:
        count = (sel["work_type"].str.lower() == wt).sum()
        label = _WORK_TYPE_LABEL.get(wt, wt.title())
        chips.append(f"<span style='background:#2d3340; border:1px solid #444; color:#ffffff; padding:2px 8px;border-radius:12px;margin-right:6px;font-size:0.85rem;'>{label}: {count}</span>")
    st.markdown(" ".join(chips), unsafe_allow_html=True)

    for wt in groups:
        block = sel[sel["work_type"].str.lower() == wt]
        if block.empty: 
            continue
        st.markdown(f"**{_WORK_TYPE_LABEL.get(wt, wt.title())}**")
        for _, r in block.sort_values(["year", "month"], ascending=[False, False]).iterrows():
            title = str(r.get("title", "Untitled")).strip()
            venue = str(r.get("venue", "")).strip()
            year  = str(r.get("year", "")).strip()
            link  = _best_link(r)
            tail  = " — ".join([x for x in [venue, year] if x])
            if link:
                st.markdown(f"- [{title}]({link}){(' — ' + tail) if tail else ''}")
            else:
                st.markdown(f"- {title}{(' — ' + tail) if tail else ''}")
    return True
