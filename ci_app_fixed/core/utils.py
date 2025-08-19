
from __future__ import annotations
import pandas as pd
import numpy as np
import ast, re
from typing import Iterable, List, Any

def format_currency(value):
    if value is None:
        return "Undisclosed"
    try:
        v = float(value)
    except Exception:
        return "Undisclosed"
    if np.isnan(v) or v <= 0:
        return "Undisclosed"
    return f"${v/1e6:.1f}M"

def parse_listish(raw: Any) -> list[str]:
    """Turn messy multi-value fields into a clean Python list."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    parts = re.split(r"[|;/]+", s)
    return [p.strip() for p in parts if p.strip()]

def unique_preserve(seq: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

_PHASE_RANK = {
    "Preclinical": 0, "Phase 1": 1, "Phase 1/2": 2, "Phase 2": 3,
    "Phase 2/3": 4, "Phase 3": 5, "Filed": 6, "Approved": 7
}
def canonical_phase(ph_list: list[str]) -> str:
    if not ph_list:
        return ""
    ph_uniq = unique_preserve(ph_list)
    ph_sorted = sorted(ph_uniq, key=lambda p: _PHASE_RANK.get(p, -1))
    return " / ".join(ph_sorted)

def parse_nct_ids(raw) -> list[str]:
    """Return a de-duplicated, order-preserving list of NCT IDs from messy inputs."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    ids = []
    if isinstance(raw, (list, tuple)):
        candidates = [str(x) for x in raw]
    else:
        s = str(raw).strip()
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
    for c in candidates:
        for m in re.findall(r"NCT\d{8}", c):
            if m not in ids:
                ids.append(m)
    return ids

def trial_links(raw) -> str:
    ids = parse_nct_ids(raw)
    if not ids:
        return "N/A"
    return " • ".join(f"[{tid}](https://clinicaltrials.gov/study/{tid})" for tid in ids)

def safe_get(dict_obj: dict, key: str, default="N/A") -> str:
    value = dict_obj.get(key, default)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value) if value else default
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return "\n".join(str(item) for item in parsed) if parsed else default
            except (ValueError, SyntaxError):
                inner = s[1:-1]
                parts = [
                    part.strip().strip("'\"")
                    for part in inner.split(",")
                    if part.strip().strip("'\"")
                ]
                return "\n".join(parts) if parts else default
    return str(value)

def norm_str(x: Any) -> str:
    return str(x).strip().lower() if pd.notna(x) else ""
