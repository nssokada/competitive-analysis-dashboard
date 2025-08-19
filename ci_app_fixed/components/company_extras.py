
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
from urllib.parse import urlparse
from core.utils import parse_listish, unique_preserve

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
        netloc = urlparse(str(url)).netloc
        return netloc.replace("www.", "") if netloc else "link"
    except Exception:
        return "link"

def _links_md(sources, max_links=6):
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

def render_company_extras_for_company(comp_row: dict, all_infos: dict) -> bool:
    """Renders funding / investors / partnerships / M&A / news / flags for a company."""
    if not all_infos:
        return False

    cid  = str(comp_row.get("company_id", "")).strip()
    name = str(comp_row.get("company_name", "")).strip()
    info_rec = all_infos.get(cid)
    if not info_rec:
        nm = str(name).strip().lower()
        for obj in all_infos.values():
            if str(obj.get("company_name","")).strip().lower() == nm:
                info_rec = obj; break
    if not info_rec:
        return False

    rendered_any = False

    # Legacy path (top-level arrays)
    if any(k in info_rec for k in ("funding_events","funding","partnerships","relationships","ma","mna","transactions","news","red_flags","yellow_flags")):
        funding = info_rec.get("funding_events") or info_rec.get("funding") or []
        if isinstance(funding, list) and funding:
            st.markdown("#### Funding history")
            dff = pd.DataFrame(funding)
            if "amount_usd" in dff.columns:
                dff["amount"] = dff["amount_usd"].apply(_fmt_money)
            show_cols = [c for c in ["date","round","amount","investors","notes"] if c in dff.columns or c == "amount"]
            st.dataframe(dff[show_cols] if show_cols else dff, use_container_width=True, hide_index=True)
            if "sources" in dff.columns:
                with st.expander("Show sources (funding)"):
                    for _, r in dff.iterrows():
                        md = _links_md(r.get("sources", []))
                        if md: st.markdown(f"- **{r.get('date','')} {r.get('round','')}** — {md}")
            rendered_any = True

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

        parts = info_rec.get("partnerships") or info_rec.get("relationships") or []
        if isinstance(parts, list) and parts:
            st.markdown("#### Partnerships & Licensing")
            dfp = pd.DataFrame(parts)
            if "value_usd" in dfp.columns:
                dfp["value"] = dfp["value_usd"].apply(_fmt_money)
            show_cols = [c for c in ["partner","type","program","announced","status","value","notes"] if c in dfp.columns or c == "value"]
            st.dataframe(dfp[show_cols] if show_cols else dfp, use_container_width=True, hide_index=True)
            if "sources" in dfp.columns:
                with st.expander("Show sources (partnerships)"):
                    for _, r in dfp.iterrows():
                        md = _links_md(r.get("sources", []))
                        if md: st.markdown(f"- **{r.get('announced','')} – {r.get('partner','')}** — {md}")
            rendered_any = True

        ma = info_rec.get("ma") or info_rec.get("mna") or info_rec.get("transactions") or []
        if isinstance(ma, list) and ma:
            st.markdown("#### M&A / Transactions")
            dfm = pd.DataFrame(ma)
            if "value_usd" in dfm.columns:
                dfm["value"] = dfm["value_usd"].apply(_fmt_money)
            show_cols = [c for c in ["role","counterparty","announced","closed","value","structure","notes"] if c in dfm.columns or c == "value"]
            st.dataframe(dfm[show_cols] if show_cols else dfm, use_container_width=True, hide_index=True)
            if "sources" in dfm.columns:
                with st.expander("Show sources (M&A)"):
                    for _, r in dfm.iterrows():
                        md = _links_md(r.get("sources", []))
                        if md: st.markdown(f"- **{r.get('announced','')} – {r.get('counterparty','')}** — {md}")
            rendered_any = True

        news = info_rec.get("news") or []
        if isinstance(news, list) and news:
            st.markdown("#### Recent news")
            try:
                news_sorted = sorted(news, key=lambda x: str(x.get("date","")), reverse=True)
            except Exception:
                news_sorted = news
            for item in news_sorted[:6]:
                title = str(item.get("title","Untitled")).strip()
                src   = str(item.get("source","")).strip()
                date  = str(item.get("date","")).strip()
                url   = item.get("url")
                st.markdown(f"- [{title}]({url}) — {src}, {date}" if url else f"- {title} — {src}, {date}")
            rendered_any = True

        flags = parse_listish(info_rec.get("red_flags")) + parse_listish(info_rec.get("yellow_flags"))
        if flags:
            st.warning("**Risks / flags:** " + "; ".join(flags))
            rendered_any = True

        return rendered_any

    # New schema path: info_rec['info'] list of events
    items = info_rec.get("info")
    if not isinstance(items, list) or not items:
        return False

    ev = pd.DataFrame(items)
    ev["__date"] = None
    for col in ["announced_date","date"]:
        if col in ev.columns:
            ev["__date"] = ev["__date"].fillna(ev[col])
    ev["__amount_usd"] = None
    for col in ["disclosed_value_usd","value_usd","amount_usd","deal_value_usd"]:
        if col in ev.columns:
            ev["__amount_usd"] = ev["__amount_usd"].fillna(ev[col])

    fund_mask = ev.get("relation_type", pd.Series([], dtype=str)).astype(str).str.lower().eq("vc_investment")
    funding_df = ev[fund_mask].copy()
    if not funding_df.empty:
        st.markdown("#### Funding history")
        g = (
            funding_df.assign(investor=funding_df.get("counterparty",""), srcs=funding_df.get("sources",None))
            .groupby([funding_df["__date"].astype(str), funding_df.get("round", pd.Series("", index=funding_df.index)).astype(str)], dropna=False)
            .agg(amount_usd=("__amount_usd","first"),
                 investors=("investor", lambda s: ", ".join(unique_preserve([x for x in s if str(x).strip()]))),
                 _sources=("srcs", lambda s: [x for row in s for x in (row or [])]))
            .reset_index()
        )
        g.columns = ["date","round","amount_usd","investors","_sources"]
        g["amount"] = g["amount_usd"].apply(_fmt_money)
        g["sources_md"] = g["_sources"].apply(_links_md)
        show_cols = [c for c in ["date","round","amount","investors"] if c in g.columns]
        st.dataframe(g[show_cols], use_container_width=True, hide_index=True)
        with st.expander("Show sources (funding)"):
            for _, r in g.iterrows():
                if r["sources_md"]:
                    st.markdown(f"- **{r['date']} {r['round']}** — {r['sources_md']}")
        invs = unique_preserve([str(x) for x in funding_df.get("counterparty", pd.Series([])).dropna().astype(str).tolist()])
        if invs:
            st.markdown("**Notable investors**")
            chips = " ".join(
                f"<span style='background:#2d3340;border:1px solid #444;color:#fff;padding:2px 8px;border-radius:12px;margin-right:6px;font-size:0.85rem;'>{i}</span>"
                for i in invs
            )
            st.markdown(chips, unsafe_allow_html=True)
        rendered_any = True
    else:
        rendered_any = False

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
        st.dataframe(out[["partner","type","program","announced","status","value","notes"]], use_container_width=True, hide_index=True)
        srcs = parts_df.get("sources", None)
        if srcs is not None:
            with st.expander("Show sources (partnerships)"):
                for _, r in parts_df.iterrows():
                    md = _links_md(r.get("sources", []))
                    if md:
                        st.markdown(f"- **{r.get('__date','')} – {r.get('counterparty','')}** — {md}")
        rendered_any = True

    ma_mask = ev.get("relation_type", pd.Series([], dtype=str)).astype(str).str.lower().isin(["acquisition","mna","merger"])
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
        st.dataframe(out[["counterparty","announced","status","value","notes"]], use_container_width=True, hide_index=True)
        srcs = ma_df.get("sources", None)
        if srcs is not None:
            with st.expander("Show sources (M&A)"):
                for _, r in ma_df.iterrows():
                    md = _links_md(r.get("sources", []))
                    if md:
                        st.markdown(f"- **{r.get('__date','')} – {r.get('counterparty','')}** — {md}")
        rendered_any = True

    return rendered_any
