import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from datetime import datetime
import networkx as nx
from itertools import combinations
import ast
import re

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Competitive Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional dark theme CSS with better contrast
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
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Session state
# -------------------------------
if 'data' not in st.session_state:
    st.session_state.data = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Overview'
if 'trials_df_norm' not in st.session_state:
    st.session_state.trials_df_norm = pd.DataFrame()

# -------------------------------
# Helpers
# -------------------------------
def load_data(uploaded_file):
    """Load and parse the uploaded JSON data"""
    if uploaded_file is not None:
        content = uploaded_file.read()
        data = json.loads(content)
        return data['data']
    return None

def format_currency(value):
    """Format currency values"""
    if pd.isna(value) or value == 0:
        return "Undisclosed"
    return f"${value/1e6:.1f}M"

def parse_listish(raw):
    """
    Turn messy multi-value fields into a clean Python list.
    Accepts real lists, tuples, or strings like "['a','b']".
    Returns [] if nothing usable.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    # try to literal_eval "[...]" shapes
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    # fallback: split on common delimiters
    parts = re.split(r"[|;/]+", s)
    return [p.strip() for p in parts if p.strip()]

def unique_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# Optional: rank phases so we can pick a canonical "highest" when needed
_PHASE_RANK = {
    "Preclinical": 0,
    "Phase 1": 1,
    "Phase 1/2": 2,
    "Phase 2": 3,
    "Phase 2/3": 4,
    "Phase 3": 5,
    "Filed": 6,
    "Approved": 7
}
def canonical_phase(ph_list):
    if not ph_list:
        return ""
    # keep unique, pick the highest phase for a single-value summary if desired
    ph_uniq = unique_preserve(ph_list)
    ph_sorted = sorted(ph_uniq, key=lambda p: _PHASE_RANK.get(p, -1))
    # pretty multi-value string for display
    return " / ".join(ph_sorted)


def parse_nct_ids(raw):
    """Return a de-duplicated, order‐preserving list of NCT IDs from messy inputs."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    ids = []

    # If it's already a list/tuple
    if isinstance(raw, (list, tuple)):
        candidates = [str(x) for x in raw]
    else:
        s = str(raw).strip()
        # If it's a string that looks like a Python list, try to parse
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    candidates = [str(x) for x in parsed]
                else:
                    candidates = [s]
            except Exception:
                candidates = [s]
        else:
            candidates = [s]

    # From all candidates, extract NCT IDs by regex
    for c in candidates:
        for m in re.findall(r"NCT\d{8}", c):
            if m not in ids:
                ids.append(m)
    return ids

def trial_links(raw):
    """Markdown links to ClinicalTrials.gov for one or more NCT IDs."""
    ids = parse_nct_ids(raw)
    if not ids:
        return "N/A"
    return " • ".join(f"[{tid}](https://clinicaltrials.gov/study/{tid})" for tid in ids)


def safe_get(dict_obj, key, default="N/A"):
    """
    Safely extract a display string:
      • None / NaN / empty list    → default
      • real list/tuple             → "\\n".join(items)
      • str that looks like [ ... ] → literal_eval → list → join
      • otherwise                   → str(value)
    """
    value = dict_obj.get(key, default)

    # 1) None / pandas NA
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default

    # 2) real list or tuple
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return "\n".join(str(item) for item in value)

    # 3) a string that *looks* like a Python list
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    if not parsed:
                        return default
                    return "\n".join(str(item) for item in parsed)
            except (ValueError, SyntaxError):
                # fallback: strip [ ], split on commas, strip quotes
                inner = s[1:-1]
                parts = [
                    part.strip().strip("'\"")
                    for part in inner.split(",")
                    if part.strip().strip("'\"")
                ]
                if parts:
                    return "\n".join(parts)
                else:
                    return default

    # 4) scalar fallback
    return str(value)

def normalize_trials(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Explode multi-NCT rows into one row per trial and assign a stable trial_key."""
    if trials_df is None or trials_df.empty:
        return pd.DataFrame()

    df = trials_df.copy()

    # Ensure string type and extract NCTs
    if 'trial_id' not in df.columns:
        df['trial_id'] = None
    df['__nct_list'] = df['trial_id'].apply(parse_nct_ids)

    # If a row has no NCT, keep it as a single row with empty NCT marker
    df['__nct_list'] = df['__nct_list'].apply(lambda x: x if x else [None])

    # One row per NCT ID
    df = df.explode('__nct_list', ignore_index=True)
    df['nct_id'] = df['__nct_list']
    df.drop(columns=['__nct_list'], inplace=True)

    # Stable trial_key: prefer the actual NCT, otherwise derive a deterministic key
    def _fallback_key(r):
        parts = [
            str(r.get('program_id', 'NA')),
            str(r.get('phase', 'NA')),
            str(r.get('trial_title', 'NA'))[:120],  # cap length to avoid huge keys
        ]
        return "||".join(parts)

    df['trial_key'] = np.where(
        df['nct_id'].notna() & (df['nct_id'].astype(str).str.len() > 0),
        df['nct_id'].astype(str),
        df.apply(_fallback_key, axis=1)
    )

    # Optional: keep a clean display label
    df['trial_label'] = np.where(
        df['nct_id'].notna() & (df['nct_id'].astype(str).str.len() > 0),
        df['nct_id'],
        df['trial_key']
    )

        # --- Clean multi-value fields for display & filters ---
    # Phase
    if 'phase' in df.columns:
        df['phase_list'] = df['phase'].apply(parse_listish)
        df['phase_clean'] = df['phase_list'].apply(canonical_phase)
    else:
        df['phase_list'] = [[] for _ in range(len(df))]
        df['phase_clean'] = ""

    # Status
    if 'status' in df.columns:
        df['status_list'] = df['status'].apply(parse_listish)
        df['status_clean'] = df['status_list'].apply(lambda xs: " / ".join(unique_preserve(xs)))
    else:
        df['status_list'] = [[] for _ in range(len(df))]
        df['status_clean'] = ""

    # Countries
    if 'countries_normalized' in df.columns:
        df['countries_list'] = df['countries_normalized'].apply(parse_listish)
        df['countries_clean'] = df['countries_list'].apply(lambda xs: " • ".join(unique_preserve(xs)))
    else:
        df['countries_list'] = [[] for _ in range(len(df))]
        df['countries_clean'] = ""

    # Title (pick first non-empty, but keep list if you want to show all)
    if 'trial_title' in df.columns:
        df['title_list'] = df['trial_title'].apply(parse_listish)
        df['trial_title_clean'] = df.apply(
            lambda r: (r['title_list'][0] if r['title_list'] else (str(r['trial_title']) if pd.notna(r['trial_title']) else "")),
            axis=1
        )
    else:
        df['title_list'] = [[] for _ in range(len(df))]
        df['trial_title_clean'] = ""


    return df


def _norm_str(x):
    return str(x).strip().lower() if pd.notna(x) else ""

def _best_link(row):
    # Prefer url; fall back to DOI link if present
    if pd.notna(row.get("url")) and str(row["url"]).strip():
        return str(row["url"]).strip()
    doi = row.get("doi")
    if pd.notna(doi) and str(doi).strip():
        return f"https://doi.org/{str(doi).strip()}"
    return None

# pretty labels + order for work types
_WORK_TYPE_LABEL = {
    "journal": "Journal articles",
    "conf_abstract": "Conference abstracts",
    "poster": "Posters",
    "preprint": "Preprints",
    "whitepaper": "Whitepapers / decks",
}
_WORK_TYPE_ORDER = ["journal", "conf_abstract", "poster", "preprint", "whitepaper"]

def render_publications_for_program(program_row, pubs_df: pd.DataFrame):
    if pubs_df is None or pubs_df.empty:
        return False

    prog_id = program_row.get("program_id")
    prog_name = _norm_str(program_row.get("program_name"))
    comp_name = _norm_str(program_row.get("company_name"))

    # 1) Primary match by program_id
    m_id = pubs_df["program_id"].astype(str).eq(str(prog_id)) if "program_id" in pubs_df.columns else pd.Series([False]*len(pubs_df))

    # 2) Fallbacks where some rows might be missing program_id but have program_name/company_name
    m_pn = pubs_df.get("program_name").apply(_norm_str).eq(prog_name) if "program_name" in pubs_df.columns else pd.Series([False]*len(pubs_df))
    m_cn = pubs_df.get("company_name").apply(_norm_str).str.contains(comp_name) if "company_name" in pubs_df.columns else pd.Series([False]*len(pubs_df))

    sel = pubs_df[m_id | (m_pn & m_cn)]

    if sel.empty:
        return False

    # Group by work_type and render
    types_present = [wt for wt in _WORK_TYPE_ORDER if wt in sel["work_type"].dropna().str.lower().unique().tolist()]
    other_types = sorted(set(sel["work_type"].dropna().str.lower()) - set(types_present))
    groups = types_present + other_types

    st.markdown("#### Publications")
    # Optional: small chips by type
    chips = []
    for wt in groups:
        count = (sel["work_type"].str.lower() == wt).sum()
        label = _WORK_TYPE_LABEL.get(wt, wt.title())
        chips.append(f"<span style='background:#2d3340; border:1px solid #444; color:#ffffff; padding:2px 8px;border-radius:12px;margin-right:6px;font-size:0.85rem;'>{label}: {count}</span>")
    st.markdown(" ".join(chips), unsafe_allow_html=True)

    # Detailed list
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

# --- Company extras: loading + rendering -----------------

def load_company_infos_blob(data: dict, extra_dir: str = None) -> dict:
    """
    Returns {company_id: info_dict}. 
    Sources:
      1) Top-level keys in the uploaded data: 'company_infos' or variants
      2) Optional local directory of JSON files named COMP_XXXX.json
    """
    infos = {}

    # 1) From the uploaded master JSON (prefer dict keyed by company_id)
    for key in ["company_infos", "company_info", "companies_extra", "companies_info"]:
        if key in data and data[key]:
            raw = data[key]
            if isinstance(raw, dict):
                for cid, obj in raw.items():
                    infos[str(cid)] = obj
            elif isinstance(raw, list):
                for obj in raw:
                    cid = str(obj.get("company_id") or obj.get("id") or "")
                    if cid:
                        infos[cid] = obj

    # 2) Optional local directory read (when running locally)
    if extra_dir:
        try:
            import os, glob, json as _json
            for p in glob.glob(os.path.join(extra_dir, "*.json")):
                with open(p, "r") as f:
                    obj = _json.load(f)
                cid = str(obj.get("company_id") or os.path.splitext(os.path.basename(p))[0])
                infos[cid] = obj
        except Exception:
            pass

    return infos


def _fmt_money(x):
    try:
        x = float(x)
        if x <= 0 or pd.isna(x):
            return "Undisclosed"
        return f"${x/1e6:.1f}M"
    except Exception:
        return "Undisclosed"

def render_company_extras_for_company(comp_row: dict, all_infos: dict) -> bool:
    """
    Renders funding / investors / partnerships / M&A / news / flags for a company.
    Supports both legacy schema (top-level funding/partnerships/ma/news) and
    the new schema where extras live under company_infos[*].info as a list of events.
    Adds clickable source URLs under each section.
    """
    import urllib.parse as _urlparse

    if not all_infos:
        return False

    cid  = str(comp_row.get("company_id", "")).strip()
    name = str(comp_row.get("company_name", "")).strip()

    # --- Locate the extras record: prefer id; fallback to normalized name
    info_rec = all_infos.get(cid)
    if not info_rec:
        nm_norm = _norm_str(name)
        for obj in all_infos.values():
            if _norm_str(obj.get("company_name")) == nm_norm:
                info_rec = obj
                break
    if not info_rec:
        return False

    # ---------- Helpers ----------
    def _fmt_money(x):
        try:
            x = float(x)
            if not np.isfinite(x) or x <= 0:
                return "Undisclosed"
            return f"${x/1e6:.1f}M"
        except Exception:
            return "Undisclosed"

    def _domain(url):
        try:
            netloc = _urlparse.urlparse(str(url)).netloc
            return netloc.replace("www.", "") if netloc else "link"
        except Exception:
            return "link"

    def _links_md(sources, max_links=6):
        """Return markdown like: [businesswire.com](url) • [fda.gov](url2) ..."""
        if not isinstance(sources, list) or not sources:
            return ""
        out = []
        for s in sources[:max_links]:
            url = s.get("url") or s.get("link")
            if not url:
                continue
            label = s.get("source_type") or _domain(url)
            out.append(f"[{label}]({url})")
        return " • ".join(out)

    rendered_any = False

    # ========== PATH A: Legacy top-level fields ==========
    if any(k in info_rec for k in ("funding_events", "funding", "partnerships", "relationships", "ma", "mna", "transactions", "news", "red_flags", "yellow_flags")):
        # ---- Funding
        funding = info_rec.get("funding_events") or info_rec.get("funding") or []
        if isinstance(funding, list) and funding:
            st.markdown("#### Funding history")
            dff = pd.DataFrame(funding)
            if "amount_usd" in dff.columns:
                dff["amount"] = dff["amount_usd"].apply(_fmt_money)
            show_cols = [c for c in ["date", "round", "amount", "investors", "notes"] if c in dff.columns or c == "amount"]
            st.dataframe(dff[show_cols] if show_cols else dff, use_container_width=True, hide_index=True)

            # Sources per row
            if "sources" in dff.columns:
                with st.expander("Show sources (funding)"):
                    for _, r in dff.iterrows():
                        md = _links_md(r.get("sources", []))
                        if md:
                            st.markdown(f"- **{r.get('date','')} {r.get('round','')}** — {md}")
            rendered_any = True

        # ---- Investors (chips)
        investors = parse_listish(info_rec.get("investors"))
        if not investors and isinstance(funding, list) and funding:
            investors = unique_preserve([i for row in funding for i in parse_listish(row.get("investors"))])
        if investors:
            st.markdown("**Notable investors**")
            chips = " ".join(
                f"<span style='background:#2d3340;border:1px solid #444;color:#fff;padding:2px 8px;border-radius:12px;margin-right:6px;font-size:0.85rem;'>{i}</span>"
                for i in investors
            )
            st.markdown(chips, unsafe_allow_html=True)
            rendered_any = True

        # ---- Partnerships
        parts = info_rec.get("partnerships") or info_rec.get("relationships") or []
        if isinstance(parts, list) and parts:
            st.markdown("#### Partnerships & Licensing")
            dfp = pd.DataFrame(parts)
            if "value_usd" in dfp.columns:
                dfp["value"] = dfp["value_usd"].apply(_fmt_money)
            show_cols = [c for c in ["partner", "type", "program", "announced", "status", "value", "notes"]
                         if c in dfp.columns or c == "value"]
            st.dataframe(dfp[show_cols] if show_cols else dfp, use_container_width=True, hide_index=True)
            if "sources" in dfp.columns:
                with st.expander("Show sources (partnerships)"):
                    for _, r in dfp.iterrows():
                        md = _links_md(r.get("sources", []))
                        if md:
                            st.markdown(f"- **{r.get('announced','')} – {r.get('partner','')}** — {md}")
            rendered_any = True

        # ---- M&A
        ma = info_rec.get("ma") or info_rec.get("mna") or info_rec.get("transactions") or []
        if isinstance(ma, list) and ma:
            st.markdown("#### M&A / Transactions")
            dfm = pd.DataFrame(ma)
            if "value_usd" in dfm.columns:
                dfm["value"] = dfm["value_usd"].apply(_fmt_money)
            show_cols = [c for c in ["role", "counterparty", "announced", "closed", "value", "structure", "notes"]
                         if c in dfm.columns or c == "value"]
            st.dataframe(dfm[show_cols] if show_cols else dfm, use_container_width=True, hide_index=True)
            if "sources" in dfm.columns:
                with st.expander("Show sources (M&A)"):
                    for _, r in dfm.iterrows():
                        md = _links_md(r.get("sources", []))
                        if md:
                            st.markdown(f"- **{r.get('announced','')} – {r.get('counterparty','')}** — {md}")
            rendered_any = True

        # ---- News
        news = info_rec.get("news") or []
        if isinstance(news, list) and news:
            st.markdown("#### Recent news")
            try:
                news_sorted = sorted(news, key=lambda x: str(x.get("date", "")), reverse=True)
            except Exception:
                news_sorted = news
            for item in news_sorted[:6]:
                title = str(item.get("title", "Untitled")).strip()
                src   = str(item.get("source", "")).strip()
                date  = str(item.get("date", "")).strip()
                url   = item.get("url")
                st.markdown(f"- [{title}]({url}) — {src}, {date}" if url else f"- {title} — {src}, {date}")
            rendered_any = True

        # ---- Flags
        flags = parse_listish(info_rec.get("red_flags")) + parse_listish(info_rec.get("yellow_flags"))
        if flags:
            st.warning("**Risks / flags:** " + "; ".join(flags))
            rendered_any = True

        return rendered_any

    # ========== PATH B: New list-of-events schema at info_rec['info'] ==========
    items = info_rec.get("info")
    if not isinstance(items, list) or not items:
        return False

    ev = pd.DataFrame(items)

    # Normalize dates/amounts
    ev["__date"] = None
    for col in ["announced_date", "date"]:
        if col in ev.columns:
            ev["__date"] = ev["__date"].fillna(ev[col])
    ev["__amount_usd"] = None
    for col in ["disclosed_value_usd", "value_usd", "amount_usd", "deal_value_usd"]:
        if col in ev.columns:
            ev["__amount_usd"] = ev["__amount_usd"].fillna(ev[col])

    # ---------- Funding (vc_investment) ----------
    fund_mask = ev.get("relation_type", pd.Series([], dtype=str)).astype(str).str.lower().eq("vc_investment")
    funding_df = ev[fund_mask].copy()
    if not funding_df.empty:
        st.markdown("#### Funding history")
        # Aggregate by date+round to combine multiple investors & sources
        g = (funding_df
             .assign(investor=funding_df.get("counterparty", ""), srcs=funding_df.get("sources", None))
             .groupby([funding_df["__date"].astype(str),
                       funding_df.get("round", pd.Series("", index=funding_df.index)).astype(str)],
                      dropna=False)
             .agg(amount_usd=("__amount_usd", "first"),
                  investors=("investor", lambda s: ", ".join(unique_preserve([x for x in s if str(x).strip()]))),
                  _sources=("srcs", lambda s: [x for row in s for x in (row or [])]))
             .reset_index())
        g.columns = ["date", "round", "amount_usd", "investors", "_sources"]
        g["amount"] = g["amount_usd"].apply(_fmt_money)
        g["sources_md"] = g["_sources"].apply(_links_md)

        show_cols = [c for c in ["date", "round", "amount", "investors"] if c in g.columns]
        st.dataframe(g[show_cols], use_container_width=True, hide_index=True)

        # Clickable sources per row
        with st.expander("Show sources (funding)"):
            for _, r in g.iterrows():
                if r["sources_md"]:
                    st.markdown(f"- **{r['date']} {r['round']}** — {r['sources_md']}")

        # Investors chips
        invs = unique_preserve([str(x) for x in funding_df.get("counterparty", pd.Series([])).dropna().astype(str).tolist()])
        if invs:
            st.markdown("**Notable investors**")
            chips = " ".join(
                f"<span style='background:#2d3340;border:1px solid #444;color:#fff;padding:2px 8px;border-radius:12px;margin-right:6px;font-size:0.85rem;'>{i}</span>"
                for i in invs
            )
            st.markdown(chips, unsafe_allow_html=True)

        rendered_any = True

    # ---------- Partnerships ----------
    part_mask = ev.get("relation_type", pd.Series([], dtype=str)).astype(str).str.lower().eq("partnership")
    parts_df = ev[part_mask].copy()
    if not parts_df.empty:
        st.markdown("#### Partnerships & Licensing")
        out = pd.DataFrame({
            "partner": parts_df.get("counterparty", pd.Series("", index=parts_df.index)),
            "type": parts_df.get("deal_type", pd.Series("", index=parts_df.index)),
            "program": parts_df.get("focus_target", pd.Series("", index=parts_df.index)),
            "announced": parts_df["__date"].astype(str),
            "status": parts_df.get("status", pd.Series("", index=parts_df.index)),
            "value_usd": parts_df["__amount_usd"],
            "notes": parts_df.get("value_notes", pd.Series("", index=parts_df.index)),
        })
        out["value"] = out["value_usd"].apply(_fmt_money)
        st.dataframe(out[["partner", "type", "program", "announced", "status", "value", "notes"]],
                     use_container_width=True, hide_index=True)

        # Sources expander
        srcs = parts_df.get("sources", None)
        if srcs is not None:
            with st.expander("Show sources (partnerships)"):
                for _, r in parts_df.iterrows():
                    md = _links_md(r.get("sources", []))
                    if md:
                        partner = r.get("counterparty", "")
                        date = str(r.get("__date", ""))
                        st.markdown(f"- **{date} – {partner}** — {md}")

        rendered_any = True

    # ---------- M&A / Transactions ----------
    ma_mask = ev.get("relation_type", pd.Series([], dtype=str)).astype(str).str.lower().isin(["acquisition", "mna", "merger"])
    ma_df = ev[ma_mask].copy()
    if not ma_df.empty:
        st.markdown("#### M&A / Transactions")
        out = pd.DataFrame({
            "counterparty": ma_df.get("counterparty", pd.Series("", index=ma_df.index)),
            "announced": ma_df["__date"].astype(str),
            "status": ma_df.get("status", pd.Series("", index=ma_df.index)),
            "value_usd": ma_df["__amount_usd"],
            "notes": ma_df.get("value_notes", pd.Series("", index=ma_df.index)),
        })
        out["value"] = out["value_usd"].apply(_fmt_money)
        st.dataframe(out[["counterparty", "announced", "status", "value", "notes"]],
                     use_container_width=True, hide_index=True)

        # Sources expander
        srcs = ma_df.get("sources", None)
        if srcs is not None:
            with st.expander("Show sources (M&A)"):
                for _, r in ma_df.iterrows():
                    md = _links_md(r.get("sources", []))
                    if md:
                        cpty = r.get("counterparty", "")
                        date = str(r.get("__date", ""))
                        st.markdown(f"- **{date} – {cpty}** — {md}")

        rendered_any = True

    return rendered_any



# -------------------------------
# Plotly theme
# -------------------------------
plotly_layout = dict(
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
        '#4C78A8',
        '#54A24B',
        '#E45756',
        '#79C36A',
        '#72B7B2',
        '#EECA3B',
        '#9B5DE5',
        '#7080A0'
    ],
    margin=dict(l=60, r=40, t=50, b=50),
    hoverlabel=dict(
        bgcolor='rgba(255,255,255,0.9)',
        font_color='#2A3F5F'
    )
)

# -------------------------------
# Sidebar navigation
# -------------------------------
with st.sidebar:
    st.markdown("## COMPETITIVE INTELLIGENCE")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    pages = ['Overview', 'Programs', 'Companies', 'Clinical Trials', 'Compare', 'Cluster Analysis']
    selected_page = st.radio("Navigation", pages, key='nav_radio')
    st.session_state.current_page = selected_page
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption("Competitive Intelligence Viz v2.0")

# =========================================================
# OVERVIEW
# =========================================================
if st.session_state.current_page == 'Overview':
    st.title("Overview")

    # File upload
    st.markdown("### Data Import")
    uploaded_file = st.file_uploader(
        "Select JSON dataset",
        type=['json', 'txt'],
        help="Upload competitive intelligence dataset in JSON format"
    )

    if uploaded_file is not None:
        st.session_state.data = load_data(uploaded_file)
        st.success("Data loaded successfully")

        # Build normalized trials and cache in session
        data = st.session_state.data or {}
        publications_df = pd.DataFrame(data.get('publications', []))
        st.session_state.publications_df = publications_df
        trials_df_raw = pd.DataFrame(data.get('trials', []))
        st.session_state.trials_df_norm = normalize_trials(trials_df_raw)

    if not st.session_state.data:
        st.info("Please upload a dataset to begin analysis")
        st.stop()

    # Base DataFrames
    data = st.session_state.data
    programs_df   = pd.DataFrame(data.get('programs', []))
    companies_df  = pd.DataFrame(data.get('companies', []))
    trials_df_raw = pd.DataFrame(data.get('trials', []))
    trials_df_norm = st.session_state.trials_df_norm if not st.session_state.trials_df_norm.empty else normalize_trials(trials_df_raw)
    partnerships  = pd.DataFrame(data.get('partnerships', []))
    st.session_state.company_infos = load_company_infos_blob(data)

    if programs_df.empty:
        st.warning("No program data found.")
        st.stop()

    # Filters
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns(4)

    def uniq(series):
        if series.name not in programs_df.columns:
            return []
        return sorted([x for x in series.dropna().unique() if str(x).strip() != ""])

    with f1:
        sel_indications = st.multiselect(
            "Indication",
            options=uniq(programs_df['indication_group']) if 'indication_group' in programs_df else [],
            default=[],
            help="Leave empty to include all"
        )
    with f2:
        sel_targets = st.multiselect(
            "Target Family",
            options=uniq(programs_df['target_family_final']) if 'target_family_final' in programs_df else [],
            default=[],
        )
    with f3:
        sel_class = st.multiselect(
            "Classification",
            options=uniq(programs_df['program_classification_final']) if 'program_classification_final' in programs_df else [],
            default=[],
        )
    with f4:
        sel_modality = st.multiselect(
            "Modality",
            options=uniq(programs_df['modality_final']) if 'modality_final' in programs_df else [],
            default=[],
        )

    # Apply filters to programs
    fdf = programs_df.copy()
    if sel_indications:
        fdf = fdf[fdf['indication_group'].isin(sel_indications)]
    if sel_targets:
        fdf = fdf[fdf['target_family_final'].isin(sel_targets)]
    if sel_class:
        fdf = fdf[fdf['program_classification_final'].isin(sel_class)]
    if sel_modality:
        fdf = fdf[fdf['modality_final'].isin(sel_modality)]

    # Derived sets for metrics
    filtered_company_names = set(fdf['company_name'].dropna().unique())
    filtered_program_ids   = set(fdf['program_id'].dropna().unique()) if 'program_id' in fdf else set()

    trials_f = pd.DataFrame()
    if not trials_df_norm.empty and 'program_id' in trials_df_norm.columns and filtered_program_ids:
        trials_f = trials_df_norm[trials_df_norm['program_id'].isin(filtered_program_ids)]

    partnerships_f = pd.DataFrame()
    if not partnerships.empty:
        col_candidates = [c for c in partnerships.columns if 'company' in c.lower() or 'partner' in c.lower()]
        if col_candidates:
            mask = False
            for c in col_candidates:
                mask = mask | partnerships[c].isin(filtered_company_names)
            partnerships_f = partnerships[mask]

    # Key Metrics (filtered)
    st.markdown("### Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Companies", len(filtered_company_names))
    with m2:
        st.metric("Active Programs", len(fdf))
    with m3:
        # use distinct trials by trial_key
        st.metric("Clinical Trials", len(trials_f.drop_duplicates(subset=['trial_key'])) if not trials_f.empty else 0)
    with m4:
        st.metric("Partnerships", len(partnerships_f))

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Overview Plots (filtered)
    st.markdown("### Therapeutic Area Distribution")
    if 'indication_group' in fdf.columns and not fdf.empty:
        ind_counts = fdf['indication_group'].value_counts().head(10)
        if not ind_counts.empty:
            fig_ind = px.bar(
                x=ind_counts.values,
                y=ind_counts.index,
                orientation='h',
                labels={'x': 'Number of Programs', 'y': 'Indication'},
                title="Leading Therapeutic Areas (filtered)"
            )
            fig_ind.update_layout(**plotly_layout, height=400)
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
                fig_stage = px.pie(
                    values=stage_counts.values,
                    names=stage_counts.index,
                    title="Development Stage Distribution (filtered)",
                    hole=0.4
                )
                fig_stage.update_layout(**plotly_layout)
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
                fig_mod = px.bar(
                    x=mod_counts.values,
                    y=mod_counts.index,
                    orientation='h',
                    title="Technology Platforms (filtered)"
                )
                fig_mod.update_layout(**plotly_layout)
                st.plotly_chart(fig_mod, use_container_width=True)
            else:
                st.info("No modalities match the current filters.")
        else:
            st.info("No modality data available.")

# =========================================================
# PROGRAMS
# =========================================================
elif st.session_state.current_page == 'Programs' and st.session_state.data:
    data = st.session_state.data
    programs_df = pd.DataFrame(data.get('programs', []))
    companies_df = pd.DataFrame(data.get('companies', []))
    trials_df_norm = st.session_state.trials_df_norm

    # Build a cached lowercase blob to search across useful fields
    _search_fields = [
        'program_name', 'company_name', 'target_primary', 'indication_primary',
        'mechanism_of_action_detailed_final', 'biological_rationale_final'
    ]
    for c in _search_fields:
        if c not in programs_df.columns:
            programs_df[c] = ""

    programs_df['_search_blob'] = (
        programs_df[_search_fields]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )


    st.title("Drug Development Programs")
    search_query = st.text_input(
        "Search programs…",
        value="",
        placeholder="Try: 'liposarcoma MDM2 Rain' or 'KRAS G12C Amgen'",
    )

    # Filters
    cols = st.columns(6)
    def _opts(df, col):
        if col in df.columns:
            return ['All'] + sorted([x for x in df[col].dropna().unique() if str(x).strip() != ""])
        return ['All']

    with cols[0]:
        indications = _opts(programs_df, 'indication_group')
        selected_indication = st.selectbox("Indication", indications)

    with cols[1]:
        targets = _opts(programs_df, 'target_family_final')
        selected_target = st.selectbox("Target Family", targets)

    with cols[2]:
        classifications = _opts(programs_df, 'program_classification_final')
        selected_classification = st.selectbox("Classification", classifications)

    with cols[3]:
        modalities = _opts(programs_df, 'modality_final')
        selected_modality = st.selectbox("Modality", modalities)

    with cols[4]:
        deliveries = _opts(programs_df, 'platform_delivery_final')
        selected_delivery = st.selectbox("Delivery", deliveries)

    with cols[5]:
        stages = _opts(programs_df, 'development_stage_final')
        selected_stage = st.selectbox("Development Stage", stages)

    # Apply filters
    filtered_df = programs_df.copy()
    if search_query.strip():
        terms = [t.lower() for t in search_query.split() if t.strip()]
        if terms:
            mask = pd.Series(True, index=filtered_df.index)
            blob = filtered_df['_search_blob']
            for t in terms:
                mask &= blob.str.contains(t, regex=False)
            filtered_df = filtered_df[mask]
    if selected_indication != 'All':
        filtered_df = filtered_df[filtered_df['indication_group'] == selected_indication]
    if selected_target != 'All':
        filtered_df = filtered_df[filtered_df['target_family_final'] == selected_target]
    if selected_classification != 'All':
        filtered_df = filtered_df[filtered_df['program_classification_final'] == selected_classification]
    if selected_modality != 'All':
        filtered_df = filtered_df[filtered_df['modality_final'] == selected_modality]
    if selected_delivery != 'All':
        filtered_df = filtered_df[filtered_df['platform_delivery_final'] == selected_delivery]
    if selected_stage != 'All':
        filtered_df = filtered_df[filtered_df['development_stage_final'] == selected_stage]

    st.markdown(f"### Results: {len(filtered_df)} programs")

    # Display programs as expandable cards
    for _, program in filtered_df.iterrows():
        company_row = companies_df[companies_df['company_id'] == program['company_id']]
        company = company_row.iloc[0].to_dict() if not company_row.empty else {}

        with st.expander(f"{program['program_name'].upper()} - {program['company_name'].title()} | {program['development_stage_final']}"):
            # Company Info Section
            st.markdown("#### Company Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**Type:** {safe_get(company, 'public_private')}")
            with col2:
                st.markdown(f"**Country:** {safe_get(company, 'country_normalized')}")
            with col3:
                st.markdown(f"**Size:** {safe_get(company, 'size_category')}")
            with col4:
                st.markdown(f"**Founded:** {safe_get(company, 'founding_year', 'N/A')}")

            st.markdown("---")

            # Program Overview
            st.markdown("#### Program Details")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Program:** {program['program_name']}")
                st.markdown(f"**Classification:** {program['program_classification_final']}")
                st.markdown(f"**Target:** {program['target_primary']}")
            with c2:
                st.markdown(f"**Indication:** {program['indication_primary']}")
                st.markdown(f"**Delivery:** {program['platform_delivery_final']}")
                st.markdown(f"**Stage:** {program['development_stage_final']}")

            # Scientific Details
            st.markdown("#### Scientific Rationale")
            st.markdown(f"**Biological Rationale:** {safe_get(program, 'biological_rationale_final')}")
            st.markdown(f"**Mechanism of Action:** {safe_get(program, 'mechanism_of_action_detailed_final')}")

            # Clinical Trials (normalized & de-duped by trial_key)
            prog_trials = pd.DataFrame()
            if not trials_df_norm.empty:
                prog_trials = trials_df_norm[trials_df_norm['program_id'] == program['program_id']]
            if not prog_trials.empty:
                st.markdown("#### Clinical Development")
                prog_trials = prog_trials.drop_duplicates(subset=['trial_key'])
                for _, tr in prog_trials.iterrows():
                    if pd.notna(tr['nct_id']) and tr['nct_id']:
                        nct_display = f"[{tr['nct_id']}](https://clinicaltrials.gov/study/{tr['nct_id']})"
                    else:
                        nct_display = "N/A"
                    st.markdown(f"- **{nct_display}**: {safe_get(tr, 'phase')} – {safe_get(tr, 'status')}")

            st.markdown(f"**Milestones:** {safe_get(program, 'timeline_milestones')}")

            # Additional Information
            st.markdown("#### Additional Information")
            st.markdown(f"**Research Notes:** {safe_get(program, 'research_notes')}")
            st.markdown(f"**Key Publications:** {safe_get(program, 'key_scientific_paper')}")
            st.markdown(f"**Data Quality Index:** {safe_get(program, 'data_quality_index')}")

            # --- Publications (by work_type) ---
            pubs_df = st.session_state.get("publications_df", pd.DataFrame())
            _ = render_publications_for_program(program, pubs_df)

            red_flags = safe_get(program, 'red_flags')
            if red_flags != 'N/A':
                st.warning(f"**Risk Factors:** {red_flags}")

# =========================================================
# COMPANIES
# =========================================================
# =========================================================
# COMPANIES
# =========================================================
elif st.session_state.current_page == 'Companies' and st.session_state.data:
    data = st.session_state.data

    # Ensure company extras are loaded even if user didn't pass through Overview this run
    if 'company_infos' not in st.session_state or not st.session_state.company_infos:
        st.session_state.company_infos = load_company_infos_blob(data or {})

    companies_df = pd.DataFrame(data.get('companies', []))
    programs_df  = pd.DataFrame(data.get('programs',  []))
    pubs_df      = st.session_state.get("publications_df", pd.DataFrame())
    trials_df_norm = st.session_state.trials_df_norm

    st.title("Companies")
    st.caption(f"Loaded extras for {len(st.session_state.get('company_infos', {}))} companies")

    if companies_df.empty:
        st.info("No company data available.")
    else:
        # derived metrics
        prog_counts = programs_df['company_name'].value_counts() if not programs_df.empty else pd.Series(dtype=int)
        companies_df['active_programs'] = companies_df['company_name'].map(prog_counts).fillna(0).astype(int)

        if 'total_funding_numeric' in companies_df.columns:
            companies_df['total_funding_numeric'] = pd.to_numeric(companies_df['total_funding_numeric'], errors='coerce')

        # filters
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            name_query = st.text_input("Search name", "")
        with c2:
            types = ['All'] + sorted([x for x in companies_df['public_private'].dropna().unique()])
            sel_type = st.selectbox("Type", types)
        with c3:
            countries = ['All'] + sorted([x for x in companies_df['country_normalized'].dropna().unique()])
            sel_country = st.selectbox("Country", countries)
        with c4:
            sizes = ['All'] + sorted([x for x in companies_df['size_category'].dropna().unique()])
            sel_size = st.selectbox("Size", sizes)

        sort_by = st.selectbox("Sort by", ["Name (A→Z)", "Active programs (↓)", "Total funding (↓)"])

        # apply filters
        fdf = companies_df.copy()
        if name_query.strip():
            nq = name_query.strip().lower()
            fdf = fdf[fdf['company_name'].fillna("").str.lower().str.contains(nq)]
        if sel_type != 'All':
            fdf = fdf[fdf['public_private'] == sel_type]
        if sel_country != 'All':
            fdf = fdf[fdf['country_normalized'] == sel_country]
        if sel_size != 'All':
            fdf = fdf[fdf['size_category'] == sel_size]

        # sort
        if sort_by == "Name (A→Z)":
            fdf = fdf.sort_values(by='company_name', ascending=True, na_position='last')
        elif sort_by == "Active programs (↓)":
            fdf = fdf.sort_values(by='active_programs', ascending=False, na_position='last')
        else:
            if 'total_funding_numeric' in fdf.columns:
                fdf = fdf.sort_values(by='total_funding_numeric', ascending=False, na_position='last')

        st.markdown(f"### Results: {len(fdf)} companies")

        # list companies as expandable cards
        for _, comp in fdf.iterrows():
            name = comp.get('company_name', 'Unknown')
            headline = f"{str(name).title()} — {safe_get(comp, 'public_private')} | {safe_get(comp, 'country_normalized')} | {int(comp.get('active_programs', 0))} programs"

            with st.expander(headline):
                # Company Information
                st.markdown("#### Company Information")
                a, b, c, d = st.columns(4)
                with a:
                    st.markdown(f"**Type:** {safe_get(comp, 'public_private')}")
                with b:
                    st.markdown(f"**Country:** {safe_get(comp, 'country_normalized')}")
                with c:
                    st.markdown(f"**Size:** {safe_get(comp, 'size_category')}")
                with d:
                    st.markdown(f"**Founded:** {safe_get(comp, 'founding_year', 'N/A')}")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    tf = comp.get('total_funding_numeric', np.nan)
                    st.metric("Total Funding", format_currency(tf if pd.notna(tf) else 0))
                with m2:
                    st.metric("Active Programs", int(comp.get('active_programs', 0)))
                with m3:
                    st.metric("HQ City", safe_get(comp, 'city_normalized'))
                with m4:
                    st.markdown(
                        f"<div style='font-size: 0.85rem; line-height: 1.2em;'><b>Primary Focus:</b><br>{safe_get(comp, 'company_focus')}</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("---")

                # Programs
                st.markdown("#### Programs")
                comp_programs = programs_df[programs_df['company_name'] == name] if not programs_df.empty else pd.DataFrame()
                if comp_programs.empty:
                    st.info("No programs found for this company.")
                else:
                    for _, program in comp_programs.iterrows():
                        with st.expander(f"{program['program_name'].upper()} | {program['development_stage_final']}"):
                            cA, cB = st.columns(2)
                            with cA:
                                st.markdown(f"**Program:** {program['program_name']}")
                                st.markdown(f"**Classification:** {program['program_classification_final']}")
                                st.markdown(f"**Target:** {program['target_primary']}")
                            with cB:
                                st.markdown(f"**Indication:** {program['indication_primary']}")
                                st.markdown(f"**Delivery:** {program['platform_delivery_final']}")
                                st.markdown(f"**Stage:** {program['development_stage_final']}")

                            st.markdown("**Biological Rationale:** " + safe_get(program, 'biological_rationale_final'))
                            st.markdown("**Mechanism of Action:** " + safe_get(program, 'mechanism_of_action_detailed_final'))

                            # Publications for this program
                            _ = render_publications_for_program(program, pubs_df)

                            # Clinical trials for this program (normalized & de-duped)
                            prog_trials = pd.DataFrame()
                            if not trials_df_norm.empty:
                                prog_trials = trials_df_norm[trials_df_norm['program_id'] == program['program_id']]
                            if not prog_trials.empty:
                                st.markdown("**Clinical Development:**")
                                for _, tr in prog_trials.drop_duplicates(subset=['trial_key']).iterrows():
                                    nct_display = f"[{tr['nct_id']}](https://clinicaltrials.gov/study/{tr['nct_id']})" if pd.notna(tr['nct_id']) and tr['nct_id'] else "N/A"
                                    st.markdown(f"- **{nct_display}**: {safe_get(tr, 'phase')} – {safe_get(tr, 'status')}")

                            red_flags = safe_get(program, 'red_flags')
                            if red_flags != 'N/A':
                                st.warning(f"**Risk Factors:** {red_flags}")

                # Company-level trials (normalized)
                st.markdown("#### Clinical Trials (Company Level)")
                comp_trials = trials_df_norm[trials_df_norm['company_name'] == name] if not trials_df_norm.empty else pd.DataFrame()
                if comp_trials.empty:
                    st.info("No company-level trials found.")
                else:
                    for _, trial in comp_trials.drop_duplicates(subset=['trial_key']).iterrows():
                        nct_display = f"[{trial['nct_id']}](https://clinicaltrials.gov/study/{trial['nct_id']})" if pd.notna(trial['nct_id']) and trial['nct_id'] else "N/A"
                        st.markdown(f"- **{nct_display}**: {safe_get(trial, 'phase')} – {safe_get(trial, 'status')}")

                # Extra notes
                st.markdown("#### Additional Information")
                st.markdown(f"**Leadership:** {safe_get(comp, 'leadership')}")
                st.markdown(f"**Website:** {safe_get(comp, 'website')}")
                st.markdown(f"**Research Notes:** {safe_get(comp, 'research_notes')}")

                # --- Company Intelligence (funding, partnerships, investors, M&A, news, flags)
                extras_shown = render_company_extras_for_company(comp.to_dict(), st.session_state.get("company_infos", {}))
                if not extras_shown and bool(comp.get("has_company_info", False)):
                    st.info("Additional company intelligence is available in your dataset but not yet loaded. "
                            "Ensure a top-level 'company_infos' is present in the uploaded JSON or provide a local extras directory.")

# =========================================================
# CLINICAL TRIALS
# =========================================================
elif st.session_state.current_page == 'Clinical Trials' and st.session_state.data:
    data = st.session_state.data
    trials_df_norm = st.session_state.trials_df_norm

    st.title("Clinical Trial Analysis")

    if trials_df_norm.empty:
        st.info("No trial data available.")
    else:
        trials_df_view = trials_df_norm.copy()

        # Build atomic option sets from the *_list columns
        phase_options = ['All']
        if 'phase_list' in trials_df_view.columns:
            phase_atomic = sorted({p for lst in trials_df_view['phase_list'] for p in (lst or [])})
            phase_options += phase_atomic

        status_options = ['All']
        if 'status_list' in trials_df_view.columns:
            status_atomic = sorted({s for lst in trials_df_view['status_list'] for s in (lst or [])})
            status_options += status_atomic

        indication_options = ['All'] + sorted(trials_df_view['indication'].dropna().unique()) if 'indication' in trials_df_view.columns else ['All']
        company_options   = ['All'] + sorted(trials_df_view['company_name'].dropna().unique()) if 'company_name' in trials_df_view.columns else ['All']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            selected_indication = st.selectbox("Indication", indication_options)
        with col2:
            selected_phase = st.selectbox("Phase", phase_options)
        with col3:
            selected_status = st.selectbox("Status", status_options)
        with col4:
            selected_company = st.selectbox("Sponsor", company_options)


        filtered_trials = trials_df_view.copy()

        if selected_indication != 'All' and 'indication' in filtered_trials.columns:
            filtered_trials = filtered_trials[filtered_trials['indication'] == selected_indication]

        if selected_phase != 'All' and 'phase_list' in filtered_trials.columns:
            filtered_trials = filtered_trials[filtered_trials['phase_list'].apply(lambda L: selected_phase in (L or []))]

        if selected_status != 'All' and 'status_list' in filtered_trials.columns:
            filtered_trials = filtered_trials[filtered_trials['status_list'].apply(lambda L: selected_status in (L or []))]

        if selected_company != 'All' and 'company_name' in filtered_trials.columns:
            filtered_trials = filtered_trials[filtered_trials['company_name'] == selected_company]

        dedup = filtered_trials.drop_duplicates(subset=['trial_key'])
        st.markdown(f"### Results: {len(dedup)} trials")


        dedup = filtered_trials.drop_duplicates(subset=['trial_key'])
        st.markdown(f"### Results: {len(dedup)} trials")

        # Display trials
        for _, trial in dedup.iterrows():
            title_left = (trial['nct_id'] if pd.notna(trial['nct_id']) and trial['nct_id'] else trial['trial_label'])
            sponsor_disp = trial['company_name'].title() if pd.notna(trial.get('company_name')) else 'Unknown'

            with st.expander(f"{title_left} - {sponsor_disp}"):
                link_source = trial.get('nct_id') or trial.get('trial_id')
                st.markdown(f"**ClinicalTrials.gov:** {trial_links(link_source)}")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Program:** {safe_get(trial, 'program_name')}")
                    # use clean fields:
                    st.markdown(f"**Status:** {trial.get('status_clean','')}")
                    st.markdown(f"**Indication:** {safe_get(trial, 'indication')}")
                with c2:
                    st.markdown(f"**Sponsor:** {safe_get(trial, 'sponsor')}")
                    st.markdown(f"**Countries:** {trial.get('countries_clean','')}")
                    st.markdown(f"**Title:** {trial.get('trial_title_clean','')}")


# =========================================================
# COMPARE
# =========================================================
elif st.session_state.current_page == 'Compare' and st.session_state.data:
    data = st.session_state.data
    programs_df = pd.DataFrame(data.get('programs', []))
    companies_df = pd.DataFrame(data.get('companies', []))
    trials_df_norm = st.session_state.trials_df_norm

    st.title("Comparative Analysis")
    tab1, tab2 = st.tabs(["Portfolio Distribution", "Company Benchmarking"])

    # ----- Tab 1: Portfolio Distribution (Programs only for clarity) -----
    with tab1:
        st.markdown("### Portfolio Distribution")

        view = st.radio(
            "View",
            ["Company × Phase", "Cross-Dimensional Matrix (original)"],
            horizontal=True
        )

        if view == "Company × Phase":
            src = st.radio("Analyze:", ["Programs"], horizontal=True)
            min_items = st.number_input("Min items per company (filter small players)", min_value=0, value=0, step=1)

            def _pick(df, name, prefs):
                for p in prefs:
                    if p in df.columns:
                        return p
                raise KeyError(f"Missing required column for {name}: tried {prefs}")

            df = programs_df.copy()
            company_col = _pick(df, "company", ["company_name"])
            phase_col   = _pick(df, "phase",   ["development_stage_final"])
            title_unit  = "programs"

            df = df[[company_col, phase_col]].copy()
            df[company_col] = df[company_col].astype(str).str.strip()
            df[phase_col]   = df[phase_col].astype(str).str.strip()

            phase_order = ["Discovery", "Preclinical", "Phase 1", "Phase 2", "Phase 3", "Filed", "Approved"]

            # aggregate
            agg = (
                df.groupby([company_col, phase_col], dropna=False)
                .size()
                .reset_index(name="count")
            )
            if agg.empty:
                st.warning("No data to visualize.")
                st.stop()

            totals = agg.groupby(company_col, as_index=False)['count'].sum().sort_values('count', ascending=False)
            if min_items > 0:
                keep = totals[totals['count'] >= min_items][company_col]
                agg  = agg[agg[company_col].isin(keep)]
                totals = totals[totals[company_col].isin(keep)]

            if totals.empty:
                st.info("All companies filtered out by the minimum threshold.")
                st.stop()

            top_default = min(10, len(totals))
            top_n = st.slider("Show top N companies by total volume", 3, min(50, len(totals)), top_default)
            top_companies = totals.head(top_n)[company_col].tolist()
            agg = agg[agg[company_col].isin(top_companies)]

            # order categories
            agg[company_col] = pd.Categorical(agg[company_col], categories=top_companies, ordered=True)
            phase_cats = [p for p in phase_order if p in agg[phase_col].unique().tolist()]
            if phase_cats:
                agg[phase_col] = pd.Categorical(
                    agg[phase_col],
                    categories=phase_cats + [p for p in agg[phase_col].unique() if p not in phase_cats],
                    ordered=True
                )

            value_col, y_label = ("count", "Count")

            # stacked bar
            fig_bar = px.bar(
                agg,
                x=company_col, y=value_col, color=phase_col,
                labels={company_col: "Company", value_col: y_label, phase_col: "Phase"},
                title=f"{title_unit.capitalize()} by Phase per Company"
            )
            fig_bar.update_layout(**plotly_layout, barmode="stack", xaxis_title=None, legend_title="Phase", height=480)
            st.plotly_chart(fig_bar, use_container_width=True)

            # heatmap
            pivot = agg.pivot_table(index=company_col, columns=phase_col, values=value_col, fill_value=0)
            pivot = pivot.reindex(index=top_companies)
            if phase_cats:
                pivot = pivot.reindex(columns=phase_cats + [c for c in pivot.columns if c not in phase_cats])

            fig_heat = px.imshow(
                pivot,
                labels=dict(x="Phase", y="Company", color=y_label),
                title=f"{title_unit.capitalize()} by Phase per Company (Heatmap)",
                aspect="auto"
            )
            layout_no_margin = {k: v for k, v in plotly_layout.items() if k != 'margin'}
            fig_heat.update_layout(
                **layout_no_margin,
                height=520,
                margin=dict(l=60, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            with st.expander("Show aggregated table"):
                st.dataframe(
                    agg.sort_values([company_col, phase_col]).rename(
                        columns={company_col: "Company", phase_col: "Phase", value_col: y_label}
                    ),
                    use_container_width=True
                )

        else:
            st.markdown("### Cross-Dimensional Analysis")
            col1, col2 = st.columns(2)
            dimensions = [
                'indication_group', 'target_family_final',
                'program_classification_final', 'modality_final',
                'development_stage_final'
            ]
            with col1:
                x_dim = st.selectbox("X-axis", dimensions, key="x_dim_matrix")
            with col2:
                y_dim = st.selectbox("Y-axis", dimensions, key="y_dim_matrix")

            if st.button("Generate Analysis", key="gen_matrix"):
                cross_tab = pd.crosstab(programs_df[y_dim], programs_df[x_dim])
                fig = px.imshow(
                    cross_tab,
                    labels=dict(x=x_dim, y=y_dim, color="Count"),
                    title="Portfolio Distribution Matrix",
                    aspect="auto",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(**plotly_layout, height=600)
                st.plotly_chart(fig, use_container_width=True)

    # ----- Tab 2: Company Benchmarking -----
    with tab2:
        st.markdown("### Company Benchmarking")
        company_names = (
            companies_df.get('company_name', pd.Series(dtype=str))
            .dropna().astype(str).unique().tolist()
        )
        company_names = sorted(company_names)

        c0, c1 = st.columns([4, 2])
        with c0:
            selected = st.multiselect(
                "Select companies (up to 6)",
                options=company_names,
                default=company_names[: min(2, len(company_names))]
            )
        with c1:
            cols_per_row = st.slider("Columns", 2, 4, 3, step=1)

        if not selected:
            st.stop()
        if len(selected) > 6:
            st.warning("Showing the first 6 selected companies.")
            selected = selected[:6]

        def _ci_eq(series, val):
            s = series.astype(str).str.strip().str.lower()
            return s == str(val).strip().lower()

        def _year_str(x):
            try:
                xi = int(float(x))
                return str(xi)
            except Exception:
                return "N/A"

        def _money(x):
            try:
                return format_currency(float(x))
            except Exception:
                return "Undisclosed"

        prog_cols = [c for c in [
            'program_name','development_stage_final','indication_primary',
            'program_classification_final','modality_final',
            'platform_delivery_final','target_primary'
        ] if c in programs_df.columns]

        trial_cols = [c for c in [
            'trial_id','phase','status','indication','enrollment_target','program_name','nct_id'
        ] if c in trials_df_norm.columns]

        def company_card(name: str):
            c = companies_df[_ci_eq(companies_df['company_name'], name)]
            cdict = c.iloc[0].to_dict() if not c.empty else {}

            progs  = programs_df[_ci_eq(programs_df['company_name'], name)] if not programs_df.empty else pd.DataFrame()
            trials = trials_df_norm[_ci_eq(trials_df_norm['company_name'], name)] if not trials_df_norm.empty else pd.DataFrame()

            # Header
            title = str(name).title()
            if len(title) > 48:
                title = title[:45] + "…"
            st.markdown(f"## {title}")

            with st.expander("Key Metrics", expanded=False):
                st.markdown(f"**Type:** {safe_get(cdict, 'public_private')}")
                st.markdown(f"**Founded:** {_year_str(cdict.get('founding_year'))}")
                st.markdown(f"**Total Funding:** {_money(cdict.get('total_funding_numeric'))}")
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
                    if trial_cols:
                        view = view[trial_cols].copy()
                    st.dataframe(view, use_container_width=True, hide_index=True)

            with st.expander("Notes", expanded=False):
                st.markdown(f"**Research Notes:** {safe_get(cdict, 'research_notes')}")

        # Grid layout
        for start in range(0, len(selected), cols_per_row):
            row = selected[start:start + cols_per_row]
            cols = st.columns(len(row))
            for i, cname in enumerate(row):
                with cols[i]:
                    company_card(cname)

# =========================================================
# CLUSTER ANALYSIS
# =========================================================
elif st.session_state.current_page == 'Cluster Analysis' and st.session_state.data:
    data = st.session_state.data
    programs_df = pd.DataFrame(data.get('programs', []))

    st.title("Competitive Landscape Network")

    if programs_df.empty:
        st.warning("No program data available")
    else:
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            primary_dimension = st.selectbox(
                "Analysis dimension",
                ["Indication Group", "Target Family"]
            )

        with col2:
            if primary_dimension == "Indication Group":
                if 'indication_group' in programs_df.columns:
                    segment_counts = programs_df['indication_group'].value_counts()
                else:
                    segment_counts = programs_df['indication_group'].value_counts()
                segment_options = [f"{seg} ({count})" for seg, count in segment_counts.items()
                                   if str(seg) != 'nan' and count >= 3]
                if segment_options:
                    selected_segment = st.selectbox("Select indication", segment_options)
                    selected_segment = selected_segment.split(' (')[0] if selected_segment else None
                else:
                    st.warning("Insufficient data for network analysis")
                    selected_segment = None
            else:
                if 'target_family_final' in programs_df.columns:
                    segment_counts = programs_df['target_family_final'].value_counts()
                else:
                    segment_counts = programs_df['target_primary'].value_counts()
                segment_options = [f"{seg} ({count})" for seg, count in segment_counts.items()
                                   if str(seg) != 'nan' and count >= 3]
                if segment_options:
                    selected_segment = st.selectbox("Select target", segment_options)
                    selected_segment = selected_segment.split(' (')[0] if selected_segment else None
                else:
                    st.warning("Insufficient data for network analysis")
                    selected_segment = None

        with col3:
            network_view = st.selectbox(
                "View type",
                ["Modality", "Platform", "Combined"]
            )

        if selected_segment:
            # Filter programs
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

            # Build network
            G = nx.Graph()
            company_info = {}
            approach_nodes = set()
            company_nodes = set()

            node_colors = {
                'Small molecule': plotly_layout['colorway'][0],
                'Antibody':        plotly_layout['colorway'][1],
                'Cell therapy':    plotly_layout['colorway'][2],
                'Gene therapy':    plotly_layout['colorway'][3],
                'RNA':             plotly_layout['colorway'][4],
                'Protein':         plotly_layout['colorway'][5],
                'Other':           plotly_layout['colorway'][6],
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
                            company_info[company] = {
                                'program_count': len(company_progs),
                                'modalities': company_progs['modality_final'].unique().tolist()
                            }
                        if modality not in approach_nodes:
                            G.add_node(modality, node_type='modality', bipartite=1)
                            approach_nodes.add(modality)
                        if G.has_edge(company, modality):
                            G[company][modality]['weight'] += 1
                        else:
                            G.add_edge(company, modality, weight=1)

            elif network_view == "Platform":
                for _, program in segment_programs.iterrows():
                    company = program['company_name']
                    platform = program['platform_delivery_final']
                    if pd.notna(company) and pd.notna(platform):
                        if company not in company_nodes:
                            G.add_node(company, node_type='company', bipartite=0)
                            company_nodes.add(company)
                            company_progs = segment_programs[segment_programs['company_name'] == company]
                            company_info[company] = {
                                'program_count': len(company_progs),
                                'platforms': company_progs['platform_delivery_final'].unique().tolist()
                            }
                        if platform not in approach_nodes:
                            G.add_node(platform, node_type='platform', bipartite=1)
                            approach_nodes.add(platform)
                        if G.has_edge(company, platform):
                            G[company][platform]['weight'] += 1
                        else:
                            G.add_edge(company, platform, weight=1)

            else:  # Combined
                for _, program in segment_programs.iterrows():
                    company = program['company_name']
                    modality = program['modality_final']
                    platform = program['platform_delivery_final']
                    if pd.notna(company) and pd.notna(modality) and pd.notna(platform):
                        approach = f"{modality} / {platform}"
                        if company not in company_nodes:
                            G.add_node(company, node_type='company', bipartite=0)
                            company_nodes.add(company)
                            company_progs = segment_programs[segment_programs['company_name'] == company]
                            company_info[company] = {'program_count': len(company_progs)}
                        if approach not in approach_nodes:
                            G.add_node(approach, node_type='approach', bipartite=1)
                            approach_nodes.add(approach)
                        if G.has_edge(company, approach):
                            G[company][approach]['weight'] += 1
                        else:
                            G.add_edge(company, approach, weight=1)

            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G, k=1.0, iterations=50, seed=42)
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
                        info = company_info.get(node, {})
                        company_sizes.append(20 + info.get('program_count', 1) * 4)

                company_trace = go.Scatter(
                    x=company_x,
                    y=company_y,
                    mode='markers+text',
                    text=company_text,
                    textposition="top center",
                    textfont=dict(size=10, color=plotly_layout['font']['color']),
                    marker=dict(
                        size=company_sizes,
                        color=plotly_layout['colorway'][0],
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
                    textfont=dict(size=11, color=plotly_layout['font']['color']),
                    marker=dict(
                        size=[25 + len(list(G.neighbors(n))) * 3 for n in approach_nodes],
                        color=[ node_colors.get(n, plotly_layout['colorway'][7]) for n in approach_nodes ],
                        symbol='diamond',
                        line=dict(width=2, color='#FFFFFF')
                    ),
                    name='Approaches',
                    hoverinfo='text',
                    hovertext=approach_text
                )

                fig = go.Figure(
                    data=edge_traces + [company_trace, approach_trace],
                    layout=go.Layout(
                        title=f'{selected_segment}: {network_view} Network',
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=20, r=20, t=60),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=700,
                        plot_bgcolor=plotly_layout['plot_bgcolor'],
                        paper_bgcolor=plotly_layout['paper_bgcolor'],
                        font=plotly_layout['font'],
                        dragmode='pan'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Companies", len(company_nodes))
                with col2:
                    st.metric("Unique Approaches", len(approach_nodes))
                with col3:
                    st.metric("Total Programs", len(segment_programs))
            else:
                st.warning("Insufficient data for network visualization")
        else:
            st.info("Select an indication or target to analyze the competitive landscape")

# =========================================================
# FALLBACK
# =========================================================
else:
    if st.session_state.current_page != 'Overview':
        st.warning("Please upload a dataset from the Overview page")
