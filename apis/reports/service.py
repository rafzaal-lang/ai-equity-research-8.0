import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.services.report.composer import compose
from src.services.financial_modeler import build_model
from src.services.macro.snapshot import macro_snapshot
from src.services.quant.signals import momentum, rsi, sma_cross
from src.services.comps.engine import comps_table
from src.services.providers import fmp_provider as fmp

app = FastAPI(title="Reports Service")

class ReportResponse(BaseModel):
    symbol: str
    markdown: str

# ROOT ENDPOINT - Fix for 404 on /
@app.get("/")
def root():
    return {
        "service": "AI Equity Research - Reports Service",
        "version": "8.0",
        "status": "healthy",
        "endpoints": {
            "health": "/v1/health",
            "report": "/v1/report/{ticker}",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

from fastapi.responses import HTMLResponse

HOME_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Equity Research</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;margin:0;background:#f7f7f8;color:#111}
    header{display:flex;gap:12px;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid #eee;background:#fff;position:sticky;top:0}
    nav a{padding:8px 14px;border-radius:999px;text-decoration:none;color:#111;border:1px solid #ddd;margin-right:8px}
    nav a.active{background:#111;color:#fff;border-color:#111}
    main{max-width:1100px;margin:20px auto;padding:0 16px;display:grid;grid-template-columns:320px 1fr;gap:16px}
    .card{background:#fff;border:1px solid #e6e6e6;border-radius:14px;padding:18px}
    .row{display:flex;gap:10px;flex-wrap:wrap}
    input,select,button,textarea{font:inherit}
    input,select{width:100%;padding:10px 12px;border:1px solid #ddd;border-radius:10px;background:#fcfcfd}
    button{padding:10px 14px;border-radius:10px;border:1px solid #111;background:#111;color:#fff;cursor:pointer}
    button.ghost{background:#fff;color:#111;border:1px solid #ddd}
    table{border-collapse:collapse;width:100%;font-size:14px}
    th,td{padding:8px;border-bottom:1px solid #eee;text-align:left}
    .pill{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #ddd;margin-right:6px;font-size:12px}
    .tag-pos{background:#0d7; color:#073}
    .tag-neg{background:#fdd; color:#a00}
    .muted{color:#777}
    pre{white-space:pre-wrap}
    .hidden{display:none}
  </style>
</head>
<body>
<header>
  <div class="row">
    <strong>Equity Research</strong>
  </div>
  <nav>
    <a id="tab-report" class="active" href="#" onclick="showTab('report');return false;">Report</a>
    <a id="tab-screens" href="#" onclick="showTab('screens');return false;">Screens</a>
    <a id="tab-retriever" href="#" onclick="showTab('retriever');return false;">Retriever</a>
  </nav>
</header>

<main>
  <!-- LEFT: inputs -->
  <section class="card" id="left-pane">
    <div class="row"><strong>Ticker</strong></div>
    <input id="ticker" value="AAPL" placeholder="e.g., AAPL"/>
    <div class="row"><small class="muted">As of (YYYY-MM-DD, optional)</small></div>
    <input id="asof" value="" placeholder="YYYY-MM-DD"/>

    <div class="row" style="margin-top:10px">
      <div style="flex:1">
        <div class="muted">Risk-free</div>
        <input id="rf" value="0.045"/>
      </div>
      <div style="flex:1">
        <div class="muted">MRP</div>
        <input id="mrp" value="0.055"/>
      </div>
      <div style="flex:1">
        <div class="muted">Debt Cost</div>
        <input id="rd" value="0.0500"/>
      </div>
    </div>

    <div style="margin:14px 0 6px">
      <label><input id="withCites" type="checkbox" checked/> Include citations (needs embeddings + filings)</label>
    </div>

    <div class="row" style="margin-top:10px">
      <button onclick="buildReport()">Build report</button>
      <button class="ghost" onclick="clearOut()">Clear</button>
    </div>

    <div style="margin-top:18px">
      <div class="muted">Providers detected:</div>
      <span class="pill" id="p-fmp">FMP</span>
      <span class="pill" id="p-sec">SEC/EDGAR</span>
      <span class="pill" id="p-qdrant">Qdrant</span>
      <span class="pill" id="p-openai">OpenAI</span>
    </div>
  </section>

  <!-- RIGHT: content -->
  <section id="content" class="card">
    <div id="pane-report">
      <h2>Quiet, deliberate analysis</h2>
      <p class="muted">Generate a one-page research note. Then switch to <b>Screens</b> for Momentum, Earnings Flash, Transcripts Q&amp;A, and Comps.</p>
      <div id="report-out" class="muted">Build a report to see output.</div>
    </div>

    <div id="pane-screens" class="hidden">
      <h2>Screens</h2>
      <div class="row" style="margin-bottom:10px">
        <button class="ghost" onclick="loadMomentum()">Momentum</button>
        <button class="ghost" onclick="loadFlash()">Earnings Flash</button>
        <button class="ghost" onclick="loadQA()">Transcripts Q&amp;A</button>
        <button class="ghost" onclick="loadComps()">Comps</button>
      </div>

      <div id="momentum-ui" class="card" style="margin:10px 0">
        <div class="row">
          <div style="flex:1">
            <div class="muted">Group</div>
            <select id="mom-group">
              <option value="subsector" selected>Subsector</option>
              <option value="sector">Sector</option>
            </select>
          </div>
          <div style="flex:1">
            <div class="muted">Windows</div>
            <input id="mom-wins" value="5d,3m,6m"/>
          </div>
          <div style="align-self:flex-end">
            <button onclick="loadMomentum()">Run</button>
          </div>
        </div>
        <div id="momentum-out" style="margin-top:12px" class="muted">No momentum yet.</div>
        <div id="drill-out" style="margin-top:12px"></div>
      </div>

      <div id="flash-out" class="card" style="margin:10px 0"></div>
      <div id="qa-out" class="card" style="margin:10px 0"></div>
      <div id="comps-out" class="card" style="margin:10px 0"></div>
    </div>

    <div id="pane-retriever" class="hidden">
      <h2>Retriever</h2>
      <p class="muted">(Your existing retriever UI can live here.)</p>
    </div>
  </section>
</main>

<script>
function showTab(name){
  for (const id of ["report","screens","retriever"]){
    document.getElementById("pane-"+id).classList.add("hidden");
    document.getElementById("tab-"+id).classList.remove("active");
  }
  document.getElementById("pane-"+name).classList.remove("hidden");
  document.getElementById("tab-"+name).classList.add("active");
}

function val(id){return document.getElementById(id).value.trim();}
function setHTML(id,html){document.getElementById(id).innerHTML = html;}

async function buildReport(){
  const t = val("ticker").toUpperCase();
  setHTML("report-out","Building…");
  try{
    const res = await fetch(`/v1/report/${encodeURIComponent(t)}?rf=${val("rf")}&mrp=${val("mrp")}&rd=${val("rd")}&asof=${encodeURIComponent(val("asof"))}&citations=${document.getElementById("withCites").checked}`);
    if(!res.ok){throw new Error(await res.text());}
    const j = await res.json();
    setHTML("report-out", `<pre>${(j.markdown||'(no markdown)')}</pre>`);
  }catch(e){
    setHTML("report-out", `<div class="muted">Error: ${e.message}</div>`);
  }
}

function clearOut(){
  setHTML("report-out","Build a report to see output.");
  setHTML("momentum-out","No momentum yet.");
  setHTML("drill-out","");
  setHTML("flash-out","");
  setHTML("qa-out","");
  setHTML("comps-out","");
}

/* ---------- Momentum ---------- */
async function loadMomentum(){
  const group = val("mom-group");
  const wins = val("mom-wins") || "5d,3m,6m";
  setHTML("momentum-out","Loading momentum…");
  try{
    const r = await fetch(`/v1/sectors/recommendations?group=${encodeURIComponent(group)}&windows=${encodeURIComponent(wins)}`);
    const j = await r.json();
    const rows = j.items || [];
    const tbl = [
      `<table><thead><tr><th>Group</th><th>5D</th><th>3M</th><th>6M</th><th>50DMA%</th><th>200DMA%</th><th></th></tr></thead><tbody>`,
      ...rows.map(it => {
        const b = it.ta?.breadth || {};
        const m = it.mom || {};
        return `<tr>
          <td>${it.name}</td>
          <td>${fmtPct(m["5d"])}</td>
          <td>${fmtPct(m["3m"])}</td>
          <td>${fmtPct(m["6m"])}</td>
          <td>${fmtPct(b.pct_above_50dma,true)}</td>
          <td>${fmtPct(b.pct_above_200dma,true)}</td>
          <td><button class="ghost" onclick="drill('${it.name}')">Drill</button></td>
        </tr>`;
      }),
      `</tbody></table>`
    ].join("");
    setHTML("momentum-out", tbl || "No data");
    setHTML("drill-out","");
  }catch(e){
    setHTML("momentum-out", `Error: ${e.message}`);
  }
}
async function drill(groupName){
  setHTML("drill-out","Loading drill-down…");
  try{
    const r = await fetch(`/v1/sectors/${encodeURIComponent(groupName)}/drilldown?show=all`);
    const j = await r.json();
    const rows = j.items || [];
    const tbl = [
      `<h3>${j.group} — drill-down</h3>`,
      `<table><thead><tr><th>Ticker</th><th>5D</th><th>3M</th><th>6M</th><th>RSI</th><th>50/200</th><th>Flag</th></tr></thead><tbody>`,
      ...rows.map(r => `<tr>
        <td>${r.ticker}</td>
        <td>${fmtPct(r.mom["5d"])}</td>
        <td>${fmtPct(r.mom["3m"])}</td>
        <td>${fmtPct(r.mom["6m"])}</td>
        <td>${fmtNum(r.ta?.rsi)}</td>
        <td>${r.ta?.dma_state?.above_50? "↑50": "↓50"} / ${r.ta?.dma_state?.above_200? "↑200":"↓200"}</td>
        <td>${r.flag}</td>
      </tr>`),
      `</tbody></table>`
    ].join("");
    setHTML("drill-out", tbl);
  }catch(e){
    setHTML("drill-out", `Error: ${e.message}`);
  }
}

/* ---------- Earnings Flash ---------- */
async function loadFlash(){
  const t = val("ticker").toUpperCase();
  setHTML("flash-out","Loading…");
  try{
    const r = await fetch(`/v1/earnings/flash/${encodeURIComponent(t)}`);
    const j = await r.json();
    const tag = j.tag === "pos" ? "tag-pos" : (j.tag === "neg" ? "tag-neg" : "");
    setHTML("flash-out", `
      <h3>Earnings Flash – ${j.ticker}</h3>
      <div><span class="pill ${tag}">${j.tag}</span> Period: ${j.period||"-"}</div>
      <ul>
        ${(j.bullets||[]).map(b=>`<li>${b}</li>`).join("")}
      </ul>
      <div class="muted">Actual EPS: ${fmtNum(j.actuals?.eps)} · Est EPS: ${fmtNum(j.consensus?.eps)} ·
      Actual Rev: ${fmtNum(j.actuals?.revenue)} · Est Rev: ${fmtNum(j.consensus?.revenue)}</div>
    `);
  }catch(e){
    setHTML("flash-out", `Error: ${e.message}`);
  }
}

/* ---------- Transcripts Q&A ---------- */
async function loadQA(){
  const t = val("ticker").toUpperCase();
  setHTML("qa-out","Loading…");
  try{
    const r = await fetch(`/v1/transcripts/qa/${encodeURIComponent(t)}`);
    const j = await r.json();
    const items = j.items||[];
    setHTML("qa-out", items.map(it=>`
      <h3>Q&A – ${j.ticker} (${it.date||""})</h3>
      <pre>${escapeHtml(it.qa_markdown||"(no Q&A)")}</pre>
    `).join("<hr/>") || "No transcripts");
  }catch(e){
    setHTML("qa-out", `Error: ${e.message}`);
  }
}

/* ---------- Comps ---------- */
async function loadComps(){
  const t = val("ticker").toUpperCase();
  setHTML("comps-out","Loading peers & table…");
  try{
    const peersRes = await fetch(`/v1/comps/${encodeURIComponent(t)}/peers`);
    const peers = await peersRes.json();
    let tbl = `<h3>Comps – ${t}</h3><div class="muted">Peers: ${(peers.peers||[]).join(", ")}</div>`;
    try{
      const tableRes = await fetch(`/v1/comps/${encodeURIComponent(t)}/table`);
      const data = await tableRes.json();
      if (data && data.rows){
        tbl += `<table><thead><tr>${
          (data.columns||[]).map(c=>`<th>${c}</th>`).join("")
        }</tr></thead><tbody>${
          data.rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join("")}</tr>`).join("")
        }</tbody></table>`;
      }
    }catch(e){}
    setHTML("comps-out", tbl);
  }catch(e){
    setHTML("comps-out", `Error: ${e.message}`);
  }
}

/* ---------- Helpers ---------- */
function fmtPct(x,alreadyPct=false){
  if (x==null||isNaN(x)) return "–";
  const v = alreadyPct ? Number(x) : (Number(x)*100);
  return (v>0?"+":"") + v.toFixed(1) + "%";
}
function fmtNum(x){
  if (x==null||isNaN(x)) return "–";
  const n = Number(x);
  if (Math.abs(n)>=1e9) return (n/1e9).toFixed(2)+"B";
  if (Math.abs(n)>=1e6) return (n/1e6).toFixed(2)+"M";
  if (Math.abs(n)>=1e3) return (n/1e3).toFixed(2)+"k";
  return n.toFixed(2);
}
function escapeHtml(s){return s.replace(/[&<>]/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[m]));}

/* default tab */
showTab('report');
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HOME_HTML


# --- Mount feature routers ---
try:
    from apis.sectors.service import router as sectors_router
    app.include_router(sectors_router, prefix="/v1")
except Exception:
    pass

try:
    from apis.earnings.service import router as earnings_router
    app.include_router(earnings_router, prefix="/v1")
except Exception:
    pass

try:
    from apis.transcripts.service import router as transcripts_router
    app.include_router(transcripts_router, prefix="/v1")
except Exception:
    pass

try:
    from apis.comps.service import router as comps_router
    app.include_router(comps_router, prefix="/v1")
except Exception:
    pass

# HEALTH CHECK ENDPOINTS - Fix for health check 404s
@app.get("/health")
def health_simple():
    return {"status": "healthy"}

@app.get("/v1/health")
def health():
    return {"status": "healthy"}

# FAVICON ENDPOINT - Fix for favicon.ico 404s
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)  # No content

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# DEBUG ENDPOINT - Check API credentials
@app.get("/debug/apis")
def debug_apis():
    import os
    return {
        "redis_url": "SET" if os.getenv('REDIS_URL') else "MISSING",
        "fmp_key": "SET" if os.getenv('FMP_API_KEY') else "MISSING", 
        "fred_key": "SET" if os.getenv('FRED_API_KEY') else "MISSING",
        "openai_key": "SET" if os.getenv('OPENAI_API_KEY') else "MISSING",
        "sec_agent": os.getenv('SEC_USER_AGENT', 'MISSING')
    }

# DEBUG ENDPOINT - Test individual APIs
@app.get("/debug/test-apis/{ticker}")
def test_individual_apis(ticker: str):
    results = {}
    
    # Test FMP API
    try:
        hist = fmp.historical_prices(ticker, limit=5)
        results["fmp_historical"] = "SUCCESS" if hist else "NO_DATA"
    except Exception as e:
        results["fmp_historical"] = f"FAILED: {str(e)}"
    
    # Test Macro (FRED)
    try:
        macro = macro_snapshot()
        results["macro_fred"] = "SUCCESS" if macro else "NO_DATA"
    except Exception as e:
        results["macro_fred"] = f"FAILED: {str(e)}"
    
    # Test OpenAI API
    try:
        import openai
        import os
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Simple test call
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        results["openai"] = "SUCCESS" if response else "NO_DATA"
    except Exception as e:
        results["openai"] = f"FAILED: {str(e)}"
    
    # Test Financial Model
    try:
        model = build_model(ticker)
        results["financial_model"] = "SUCCESS" if model and "error" not in model else "NO_DATA"
    except Exception as e:
        results["financial_model"] = f"FAILED: {str(e)}"
    
    # Test Comps
    try:
        comp = comps_table(ticker)
        results["comps"] = "SUCCESS" if comp else "NO_DATA"
    except Exception as e:
        results["comps"] = f"FAILED: {str(e)}"
    
    return results

@app.get("/v1/report/{ticker}", response_model=ReportResponse)
def get_report(ticker: str):
    try:
        # Gather all data
        model = build_model(ticker)
        if "error" in model:
            raise HTTPException(status_code=404, detail=model["error"])
        
        macro = macro_snapshot()
        hist = fmp.historical_prices(ticker, limit=300)
        q = {
            "momentum": momentum(hist),
            "rsi": rsi(hist),
            "sma": sma_cross(hist),
        }
        comp = comps_table(ticker)
        
        # Compose report
        md = compose({
            "symbol": ticker.upper(),
            "as_of": macro.get("as_of"),
            "call": "Review", "conviction": 6.5, "target_low": "—", "target_high": "—",
            "macro": macro,
            "quant": {"momentum": q["momentum"], "rsi": q["rsi"], "sma": q["sma"]},
            "fundamentals": model["core_financials"],
            "dcf": model["dcf_valuation"],
            "comps": {"peers": comp["peers"][:10]},
        })
        return ReportResponse(symbol=ticker.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8086)
