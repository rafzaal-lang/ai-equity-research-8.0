import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fastapi import FastAPI
from src.services.macro.snapshot import macro_snapshot

app = FastAPI(title="Macro Service")

@app.get("/v1/health")
def health():
    return {"status": "healthy"}

@app.get("/v1/macro")
def get_macro():
    return macro_snapshot()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)

