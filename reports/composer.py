# reports/composer.py
from __future__ import annotations
from typing import Any, Dict, List

def _fmt_money(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "—"

def _fmt_pct(x):
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"

def _fmt_num(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

def compose(payload: Dict[str, Any]) -> str:
    """
    Expect keys (all optional, handled gracefully):
      symbol, as_of, base_currency
      fundamentals: dict (e.g., revenue, ebit, net_income, gross_margin, op_margin, fcf)
      valuation: dict with wacc
      dcf: dict or None
      comps: {"peers": [ {ticker, pe, ps, ev_ebitda, fcf_yield, market_cap}, ... ]}
      citations: [ {title, source_type, date, url}, ... ]
    """
    sym = (payload.get("symbol") or "").upper()
    as_of = payload.get("as_of") or "latest"
    base_ccy = payload.get("base_currency", "USD")

    f = payload.get("fundamentals") or {}
    wacc = (payload.get("valuation") or {}).get("wacc")
    dcf = payload.get("dcf") or {}
    peers: List[Dict[str, Any]] = (payload.get("comps") or {}).get("peers") or []
    citations = payload.get("citations") or []

    # --- Header ---
    out = []
    out.append(f"# {sym} — Equity Research Note")
    out.append(f"*As of:* `{as_of}`  •  *Base currency:* `{base_ccy}`")
    out.append("")

    # --- Summary ---
    out.append("## Summary")
    summary_lines = []
    if wacc is not None:
        summary_lines.append(f"- Estimated WACC: **{_fmt_pct(wacc)}**")
    if f.get("revenue") is not None:
        summary_lines.append(f"- Revenue (TTM): **{_fmt_money(f.get('revenue'))}**")
    if f.get("net_income") is not None:
        summary_lines.append(f"- Net Income (TTM): **{_fmt_money(f.get('net_income'))}**")
    if dcf:
        base = dcf.get("base_value")
        if base is not None:
            summary_lines.append(f"- DCF Base Value: **{_fmt_money(base)}**")
    if not summary_lines:
        summary_lines.append("- Summary unavailable (insufficient data).")
    out.extend(summary_lines)
    out.append("")

    # --- Fundamentals table ---
    out.append("## Fundamentals (TTM / latest)")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---:|")
    rows = [
        ("Revenue", _fmt_money(f.get("revenue"))),
        ("EBIT", _fmt_money(f.get("ebit"))),
        ("Net Income", _fmt_money(f.get("net_income"))),
        ("Free Cash Flow", _fmt_money(f.get("fcf"))),
        ("Gross Margin", _fmt_pct(f.get("gross_margin"))),
        ("Operating Margin", _fmt_pct(f.get("op_margin"))),
        ("FCF Margin", _fmt_pct(f.get("fcf_margin"))),
        ("ROIC", _fmt_pct(f.get("roic"))),
        ("Debt/Equity", _fmt_num(f.get("de_ratio"))),
    ]
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    out.append("")

    # --- Valuation / WACC ---
    out.append("## Valuation")
    if wacc is not None:
        out.append(f"- **WACC:** {_fmt_pct(wacc)}")
    else:
        out.append("- **WACC:** —")
    if dcf:
        base = _fmt_money(dcf.get("base_value"))
        low = _fmt_money(dcf.get("low_value"))
        high = _fmt_money(dcf.get("high_value"))
        out.append(f"- **DCF Value (Low / Base / High):** {low} / {base} / {high}")
    out.append("")

    # --- Comps table ---
    out.append("## Peers (selected)")
    out.append("")
    if peers:
        out.append("| Ticker | P/E | P/S | EV/EBITDA | FCF Yield | Market Cap |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for r in peers[:20]:
            out.append(
                f"| {r.get('ticker','—')} | "
                f"{_fmt_num(r.get('pe'))} | "
                f"{_fmt_num(r.get('ps'))} | "
                f"{_fmt_num(r.get('ev_ebitda'))} | "
                f"{_fmt_pct(r.get('fcf_yield'))} | "
                f"{_fmt_money(r.get('market_cap'))} |"
            )
    else:
        out.append("_No comps available._")
    out.append("")

    # --- Citations ---
    if citations:
        out.append("## Citations")
        for c in citations:
            title = c.get("title") or "document"
            date = c.get("date") or "—"
            src = c.get("source_type") or "EDGAR"
            url = c.get("url")
            if url:
                out.append(f"- **{title}** ({src}, {date}) — {url}")
            else:
                out.append(f"- **{title}** ({src}, {date})")
        out.append("")

    return "\n".join(out)
