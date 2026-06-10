from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.routers import forecast, analytics, supply_chain, pricing
from app.utils.model_registry import models_ready

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Dark Store Forecast API Starting ===")
    if models_ready():
        logger.info("Pre-trained models found. Loading from disk.")
    else:
        logger.warning(
            "Models not found in models/ directory. "
            "Run scripts/train_and_pickle.py to generate them. "
            "Forecast endpoints will return empty results until models are available."
        )
    yield
    logger.info("=== Dark Store Forecast API Shutting Down ===")


app = FastAPI(
    title="Dark Store Supply Chain & Demand Forecast API",
    description=(
        "API for Delhi NCR dark store perishables forecasting. "
        "Provides demand forecasts (daily/weekly/monthly), "
        "supply chain planning, inventory risk analysis, and pricing insights."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router, prefix="/forecast", tags=["Forecasting"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
app.include_router(supply_chain.router, prefix="/supply-chain", tags=["Supply Chain"])
app.include_router(pricing.router, prefix="/pricing", tags=["Pricing"])


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "models_ready": models_ready(),
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "models_ready": models_ready()}