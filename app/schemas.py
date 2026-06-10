from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import date


class ForecastRequest(BaseModel):
    product: str
    horizon_days: int = Field(default=30, ge=1, le=365)
    model_type: str = Field(default="lgbm", pattern="^(lgbm|prophet)$")
    freq: str = Field(default="D", pattern="^(D|W|ME|QE|YE)$")


class ForecastPoint(BaseModel):
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    product: str
    product_label: str
    model_type: str
    freq: str
    forecast: List[ForecastPoint]


class AnalyticsRequest(BaseModel):
    start_date: date
    end_date: date
    products: Optional[List[str]] = None
    granularity: str = Field(default="D", pattern="^(D|W|ME|QE)$")


class LiabilityCosts(BaseModel):
    transport: float = 0.0
    last_mile: float = 0.0
    manpower: float = 0.0
    storage_electricity: float = 0.0
    rent_licences: float = 0.0
    miscellaneous: float = 0.0


class VendorPricing(BaseModel):
    vendor_name: str
    prices: Dict[str, Optional[float]]  # product_col -> price per kg


class RestockRequest(BaseModel):
    products: List[str]
    shelf_life_days: Dict[str, int]
    safety_stock_multiplier: float = 1.2
    lead_time_days: int = 1
    vendor_cost_per_kg: Dict[str, float]
    selling_price_per_kg: Dict[str, float]
    liability_costs: LiabilityCosts
    horizon_days: int = 7


class PricingAnalysisRequest(BaseModel):
    products: List[str]
    your_price: Dict[str, float]           # product_col -> selling price
    competitor_prices: Dict[str, float]    # product_col -> competitor price
    vendor_cost: Dict[str, float]          # product_col -> procurement cost
    unit_type: Dict[str, str]              # product_col -> "kg"|"grams"|"units"
    unit_weight_grams: Dict[str, float]    # only relevant if unit_type == "units"
    forecast_monthly_kg: Dict[str, float]  # product_col -> forecasted monthly demand in kg