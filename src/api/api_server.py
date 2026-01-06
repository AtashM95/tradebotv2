
from fastapi import FastAPI
from .endpoints.trading_endpoints import router as trading_router
from .endpoints.health_endpoints import router as health_router

app = FastAPI()
app.include_router(trading_router, prefix='/trading')
app.include_router(health_router, prefix='/health')
