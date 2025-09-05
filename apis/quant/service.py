import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fastapi import FastAPI, HTTPException
from src.services.quant.signals import momentum, rsi, sma_cross, atr
from src.services.providers import fmp_provider as fmp

app = FastAPI(title="Quant Service")

@app.get("/v1/health")
def health():
    return {"status": "healthy"}

@app.get("/v1/quant/{ticker}")
def get_quant(ticker: str):
    try:
        hist = fmp.historical_prices(ticker, limit=300)
        if not hist:
            raise HTTPException(status_code=404, detail="No price history found")
        
        return {
            "symbol": ticker.upper(),
            "momentum": momentum(hist),
            "rsi": rsi(hist),
            "sma": sma_cross(hist),
            "atr": atr(hist),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)

