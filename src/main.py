"""TingWu Speech Service 主入口"""
from fastapi import FastAPI

app = FastAPI(title="TingWu Speech Service", version="1.0.0")

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/")
async def root():
    return {"name": "TingWu Speech Service", "version": "1.0.0"}
