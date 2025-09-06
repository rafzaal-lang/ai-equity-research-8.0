# ui_minimal.py
# Minimal, production-friendly FastAPI UI.
# - Keeps imports light at module load (heavy deps are lazy-imported inside routes)
# - Works on Render or Docker: start with `uvicorn ui_minimal:app --host 0.0.0.0 --port $PORT`

from __future__ import annotations
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from jinja2 import Environment, BaseLoader, select_autoescape

# ---- Fast, safe startup: define `app` immediately ----
app = FastAPI(title="Equity Research — Minimal UI")

@app.get("/health")
def health():
    return {"ok": True}

# ---- Light templating (inline Jinja) ----
BASE_CURRENCY = os.getenv("BASE_CURRENCY", "USD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

BASE_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Equity Research</title>
    <style>
      :root { --bg:#f7f7f5; --panel:#fff; --ink:#0a0a0a; --muted:#6b7280; --line:#e5e7eb; --radius:16px; }
      *{box-sizing:border-box} html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
        font-family: ui-sans-serif,-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Helvetica,Arial,"Segoe UI",Roboto,"Noto Sans",sans-serif}
      a{color:#0f172a;text-decoration:none} a:hover{text-decoration:underline}
      .container{max-width:1100px;margin:0 auto;padding:24px}
      header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
      .brand{font-weight:700;letter-spacing:.3px}
      .nav a{margin-left:12px;padding:8px 12px;border:1px solid var(--line);border-radius:999px;background:#fff}
      .nav a.active{background:var(--ink);color:#fff;border-color:var(--ink)}
      .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px}
      .grid{display:grid;gap:16px} @media (min-width:960px){.grid-3{grid-template-columns:320px 1fr}}
      .label{font-size:12px;color:var(--muted);margin-bottom:6px}
      .input,.number{width:100%;padding:10px 12px;border-radius:12px;border:1px solid var(--line);background:#fff}
      .row{display:grid;gap:10px;grid-template-columns:repeat(3,1fr)}
      .btn{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:12px;border:1px solid var(--ink);color:#fff;background:var(--ink);cursor:pointer}
      .btn.secondary{background:#fff;color:var(--ink);border-color:var(--line)}
      .muted{color:var(--muted)}
      .cards{display:grid;gap:10px;grid-template-columns:repeat(2,1fr)} @media (min-width:960px){.cards{grid-template-columns:repeat(4,1fr)}}
      .card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px}
      .k{font-size:12px;color:var(--muted);margin-bottom:2px} .v{font-size:18px;font-weight:600}
      table{width:100%;border-collapse:collapse} th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left;font-size:14px}
      th{color:var(--muted);font-weight:600} .pill{display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:999px;font-size:12px;color:var(--muted)}
      .md{line-height:1.6} .md h1,.md h2,.md h3{margin-top:1.2em}
      footer{margin-top:26px;font-size:12px;color:var(--muted)}
    </style>
  </head>
  <body>
    <div class="container">
      <header>
        <div class="brand">Equity Research</div>
        <nav class="nav">
          <a href="/" class="{{ 'active' if active=='report' else '' }}">Report</a>
          <a href="/screens" class="{{ 'active' if active=='screens' else '' }}">Screens</a>
          <a href="/retriever" class="{{ 'active' if active=='retriever' else '' }}">Retriever</a>
        </nav>
      </header>
      {{ content | safe }}
      <footer>Base Currency: {{ base_currency }} · Minimal UI</footer>
    </div>
  </body>
</html>
"""

REPORT_FORM = """
<div class="grid grid-3">
  <div class="panel">
    <form method="post" action="/report">
      <div class="label">Ticker</div>
      <input class="input" type="text" name="ticker" placeholder="AAPL" required />
      <div class="label" style="margin-top:10px;">As of (YYYY-MM-DD, optional)</div>
      <input class="input" type="text" name="as_of" placeholder="2025-06-30" />
      <div class="row" style="margin-top:10px;">
        <div>
          <div class="label">Risk-free</div>
          <input class="number" type="number" step="0.0001" name="rf" value="0.045" />
        </div>
        <div>
          <div class="label">MRP</div>
          <input class="number" type="number" step="0.0001" name="mrp" value="0.055" />
        </div>
        <div>
          <div class="label">Debt Cost</div>
          <input class="number" type="number" step="0.0001" name="kd" value="0.0500" />
        </div>
      </div>
      <label style="display:flex;align-items:center;gap:8px;margin-top:10px;">
        <input type="checkbox" name="include_citations" />
        <span class="muted">Include citations (needs embeddings + filings)</span>
      </label>
      <div style="margin-top:12px;">
        <button class="btn" type="submit">Build report</button>
      </div>
    </form>
  </div>
  <div class="panel" id="hero">
    <div class="muted">Welcome</div>
    <h2 style="margin:.2rem 0 1rem 0;">Quiet, deliberate analysis</h2>
    <p class="muted">Generate a one-page research note with fundamentals, a peer-beta WACC and a DCF. Add citations if you have EDGAR chunks in your vector store.</p>
    <div style="margin-top:12px;">
      <span class="pill">FMP</span> <span class="pill">SEC/EDGAR</span> <span class="pill">Qdrant</span> <span class="pill">OpenAI</span>
    </div>
  </div>
</div>
"""

REPORT_RESULT = """
<div class="grid" style="margin-top:16px;">
  <div class="cards">
    <div class="card"><div class="k">Market Cap</div><div class="v">{{ kpis.market_cap }}</div></div>
    <div class="card"><div class="k">Enterprise Value</div><div class="v">{{ kpis.ev }}</div></div>
    <div class="card"><div class="k">P/E</div><div class="v">{{ kpis.pe }}</div></div>
    <div class="card"><div class="k">FCF Yield</div><div class="v">{{ kpis.fcf_yield }}</div></div>
  </div>
  <div class="panel md" style="margin-top:12px;">
    {{ report_html | safe }}
    <div style="margin-top:12px; display:flex; gap:8px;">
      <a class="btn secondary" href="/report.md?ticker={{ ticker }}&as_of={{ as_of or '' }}&rf={{ rf }}&mrp={{ mrp }}&kd={{ kd }}&cit={{ '1' if include_citations else '0' }}">Download Markdown</a>
    </div>
  </div>
</div>
"""

SCREENS_PAGE = """
<div class="panel">
  <form method="post" action="/screens">
    <div class="row">
      <div>
        <div class="label">Base Ticker</div>
        <input class="input" type="text" name="base_ticker" placeholder="AAPL" required />
      </div>
      <div>
        <div class="label">Min Market Cap (USD)</div>
        <input class="number" type="number" name="size_min" placeholder="10000000000" />
      </div>
      <div>
        <div class="label">As of (optional)</div>
        <input class="input" type="text" name="as_of" placeholder="2025-06-30" />
      </div>
    </div>
    <div style="margin-top:12px;">
      <button class="btn" type="submit">Run screen</button>
    </div>
  </form>
</div>
{% if rows %}
<div class="panel" style="margin-top:12px;">
  <div class="muted" style="margin-bottom:8px;">Top 20 (by composite value score)</div>
  <table>
    <thead>
      <tr>
        <th>Rank</th><th>Ticker</th><th>P/E</th><th>P/S</th><th>EV/EBITDA</th><th>FCF Yield</th><th>Market Cap</th>
      </tr>
    </thead>
    <tbody>
    {% for r in rows[:20] %}
      <tr>
        <td>{{ r.value_rank }}</td>
        <td>{{ r.ticker }}</td>
        <td>{{ '%.2f'|format(r.pe) if r.pe is not none else '—' }}</td>
        <td>{{ '%.2f'|format(r.ps) if r.ps is not none else '—' }}</td>
        <td>{{ '%.2f'|format(r.ev_ebitda) if r.ev_ebitda is not none else '—' }}</td>
        <td>{{ '%.2f%%'|format((r.fcf_yield or 0)*100) if r.fcf_yield is not none else '—' }}</td>
        <td>{{ '{:,.0f}'.format(r.market_cap) if r.market_cap else '—' }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}
"""

RETRIEVER_PAGE = """
<div class="panel">
  <form method="post" action="/retriever">
    <div class="label">Query</div>
    <input class="input" type="text" name="query" placeholder="risk factors supply chain disruptions" required />
    <div class="label" style="margin-top:10px;">Tickers (optional, comma-separated)</div>
    <input class="input" type="text" name="tickers" placeholder="AAPL,MSFT" />
    <div style="margin-top:12px;">
      <button class="btn" type="submit">Search filings</button>
      <span class="muted" style="margin-left:8px;">Embeddings required: {{ 'Yes' if has_openai else 'No' }}</span>
    </div>
  </form>
</div>
{% if hits %}
<div class="panel" style="margin-top:12px;">
  <div class="muted" style="margin-bottom:8px;">Top results</div>
  <table>
    <thead><tr><th>Score</th><th>Ticker</th><th>Type</th><th>Date</th><th>Section</th><th>Link</th></tr></thead>
    <tbody>
    {% for h in hits %}
      <tr>
        <td>{{ '%.3f'|format(h.score) }}</td>
        <td>{{ h.payload.ticker or '—' }}</td>
        <td>{{ h.payload.source_type or '—' }}</td>
        <td>{{ h.payload.filing_date or '—' }}</td>
        <td>{{ h.payload.section or '—' }}</td>
        <td>{% if h.payload.source_url %}<a href="{{ h.payload.source_url }}" target="_blank">Open</a>{% else %}—{% endif %}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}
"""

env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
def render(page: str, **kw) -> str:
    tpl = env.from_string(BASE_HTML)
    content_tpl = env.from_string(page)
    return tpl.render(content=content_tpl.render(**kw), active=kw.get("active", "report"), base_currency=BASE_CURRENCY)

# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(render(REPORT_FORM, active="report"))

@app.post("/report", response_class=HTMLResponse)
def post_report(
    ticker: str = Form(...),
    as_of: Optional[str] = Form(None),
    rf: float = Form(0.045),
    mrp: float = Form(0.055),
    kd: float = Form(0.05),
    include_citations: Optional[str] = Form(None),
):
    # Lazy imports to avoid import-time failures
    import markdown as md
    from src.services.financial_modeler import build_model
    from src.services.comps.engine import comps_table, latest_metrics
    from src.services.wacc.peer_beta import peer_beta_wacc
    from src.services.report.composer import compose as compose_report

    # Build core model
    model = build_model(ticker, force_refresh=False)
    if "error" in model:
        msg = f'<p class="muted" style="margin-top:10px;">Error: {model["error"]}</p>'
        return HTMLResponse(render(REPORT_FORM + msg, active="report"), status_code=400)

    comps = comps_table(ticker, as_of=as_of, max_peers=25)
    peers = [r["ticker"] for r in comps["peers"][1:]]
    w = peer_beta_wacc(ticker, peers, rf=rf, mrp=mrp, kd=kd, target_d_e=None, as_of=as_of)

    # Optional citations via embeddings + qdrant
    citations: List[dict] = []
    if include_citations and OPENAI_API_KEY:
        try:
            from openai import OpenAI
            from src.services.vector_service import search as vec_search
            from src.services.ranking_service import final_score
            client = OpenAI(api_key=OPENAI_API_KEY)
            emb = client.embeddings.create(model=EMBED_MODEL, input=f"{ticker} latest 10-K 10-Q risk factors management discussion").data[0].embedding
            raw_hits = vec_search(emb, k=8, tickers=[ticker.upper()])
            rescored = []
            for h in raw_hits:
                p = h.get("payload", {}) or {}
                s = final_score(h["score"], p.get("filing_date"), p.get("source_type"),
                                [p.get("ticker")] if p.get("ticker") else None, [ticker.upper()])
                rescored.append((s, p))
            rescored.sort(key=lambda x: x[0], reverse=True)
            for _, p in rescored[:5]:
                citations.append({
                    "title": p.get("section", "document"),
                    "source_type": p.get("source_type"),
                    "date": p.get("filing_date"),
                    "url": p.get("source_url"),
                })
        except Exception:
            # Silent fail; UI still renders report
            pass

    md_text = compose_report({
        "symbol": ticker.upper(),
        "as_of": as_of or "latest",
        "call": "Review",
        "conviction": 7.0,
        "target_low": "—",
        "target_high": "—",
        "base_currency": BASE_CURRENCY,
        "fundamentals": model["core_financials"],
        "dcf": model.get("dcf_valuation"),
        "valuation": {"wacc": w.get("wacc")},
        "comps": {"peers": comps["peers"]},
        "citations": citations,
        "artifact_id": "ui-session",
    })
    report_html = md.markdown(md_text, extensions=["tables"])

    lm = latest_metrics(ticker, as_of=as_of)
    def fmt_money(x): return f"${x:,.0f}" if isinstance(x, (int, float)) and x is not None else "—"
    def fmt_pct(x): return f"{x*100:.2f}%" if isinstance(x, (int, float)) and x is not None else "—"
    kpis = {
        "market_cap": fmt_money(lm.get("market_cap")),
        "ev": fmt_money(lm.get("enterprise_value")),
        "pe": f"{lm.get('pe'):.2f}" if lm.get("pe") is not None else "—",
        "fcf_yield": fmt_pct(lm.get("fcf_yield")),
    }

    return HTMLResponse(render(REPORT_RESULT, active="report",
                               report_html=report_html, ticker=ticker.upper(), as_of=as_of,
                               rf=rf, mrp=mrp, kd=kd, include_citations=bool(include_citations), kpis=kpis))

@app.get("/report.md", response_class=PlainTextResponse)
def download_report_md(
    ticker: str,
    as_of: Optional[str] = None,
    rf: float = 0.045,
    mrp: float = 0.055,
    kd: float = 0.05,
    cit: str = "0",
):
    from src.services.financial_modeler import build_model
    from src.services.comps.engine import comps_table
    from src.services.wacc.peer_beta import peer_beta_wacc
    from src.services.report.composer import compose as compose_report

    model = build_model(ticker, force_refresh=False)
    if "error" in model:
        return PlainTextResponse(f"Error: {model['error']}", status_code=400)

    comps = comps_table(ticker, as_of=as_of, max_peers=25)
    peers = [r["ticker"] for r in comps["peers"][1:]]
    w = peer_beta_wacc(ticker, peers, rf=rf, mrp=mrp, kd=kd, target_d_e=None, as_of=as_of)

    citations: List[dict] = []
    if cit == "1" and OPENAI_API_KEY:
        try:
            from openai import OpenAI
            from src.services.vector_service import search as vec_search
            from src.services.ranking_service import final_score
            client = OpenAI(api_key=OPENAI_API_KEY)
            emb = client.embeddings.create(model=EMBED_MODEL, input=f"{ticker} latest 10-K 10-Q risk factors management discussion").data[0].embedding
            raw_hits = vec_search(emb, k=8, tickers=[ticker.upper()])
            rescored = []
            for h in raw_hits:
                p = h.get("payload", {}) or {}
                s = final_score(h["score"], p.get("filing_date"), p.get("source_type"),
                                [p.get("ticker")] if p.get("ticker") else None, [ticker.upper()])
                rescored.append((s, p))
            rescored.sort(key=lambda x: x[0], reverse=True)
            for _, p in rescored[:5]:
                citations.append({
                    "title": p.get("section", "document"),
                    "source_type": p.get("source_type"),
                    "date": p.get("filing_date"),
                    "url": p.get("source_url"),
                })
        except Exception:
            pass

    md_text = compose_report({
        "symbol": ticker.upper(),
        "as_of": as_of or "latest",
        "call": "Review",
        "conviction": 7.0,
        "target_low": "—",
        "target_high": "—",
        "base_currency": BASE_CURRENCY,
        "fundamentals": model["core_financials"],
        "dcf": model.get("dcf_valuation"),
        "valuation": {"wacc": w.get("wacc")},
        "comps": {"peers": comps["peers"]},
        "citations": citations,
        "artifact_id": "ui-session",
    })
    return PlainTextResponse(md_text)

@app.get("/screens", response_class=HTMLResponse)
def screens_get():
    return HTMLResponse(render(SCREENS_PAGE, active="screens", rows=[]))

@app.post("/screens", response_class=HTMLResponse)
def screens_post(
    base_ticker: str = Form(...),
    size_min: Optional[float] = Form(None),
    as_of: Optional[str] = Form(None),
):
    from src.services.comps.engine import comps_table
    data = comps_table(base_ticker, as_of=as_of, max_peers=50)
    rows = data["peers"]
    if size_min is not None:
        rows = [r for r in rows if (r.get("market_cap") or 0) >= size_min]
    return HTMLResponse(render(SCREENS_PAGE, active="screens", rows=rows))

@app.get("/retriever", response_class=HTMLResponse)
def retriever_get():
    return HTMLResponse(render(RETRIEVER_PAGE, active="retriever", hits=[], has_openai=bool(OPENAI_API_KEY)))

@app.post("/retriever", response_class=HTMLResponse)
def retriever_post(query: str = Form(...), tickers: Optional[str] = Form(None)):
    if not OPENAI_API_KEY:
        note = '<p class="muted" style="margin-top:10px;">OpenAI API key not set; cannot embed query.</p>'
        return HTMLResponse(render(RETRIEVER_PAGE + note, active="retriever", hits=[], has_openai=False), status_code=400)

    try:
        from openai import OpenAI
        from src.services.vector_service import search as vec_search
        from src.services.ranking_service import final_score

        client = OpenAI(api_key=OPENAI_API_KEY)
        emb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
        tickers_list = [t.strip().upper() for t in tickers.split(",")] if tickers else None
        raw_hits = vec_search(emb, k=12, tickers=tickers_list)
        rescored = []
        for h in raw_hits:
            p = h.get("payload", {}) or {}
            s = final_score(
                h["score"],
                p.get("filing_date"),
                p.get("source_type"),
                [p.get("ticker")] if p.get("ticker") else None,
                tickers_list,
            )
            rescored.append({"score": s, "payload": p})
        rescored.sort(key=lambda x: x["score"], reverse=True)
        return HTMLResponse(render(RETRIEVER_PAGE, active="retriever", hits=rescored[:8], has_openai=True))
    except Exception as e:
        note = f'<p class="muted" style="margin-top:10px;">Error: {str(e)}</p>'
        return HTMLResponse(render(RETRIEVER_PAGE + note, active="retriever", hits=[], has_openai=True), status_code=500)

