import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
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

@app.get("/v1/health")
def health():
    return {"status": "healthy"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

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

