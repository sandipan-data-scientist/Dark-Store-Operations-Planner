import joblib
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any

from app.config import MODELS_DIR, ALL_PRODUCTS, PRETTY_NAMES

logger = logging.getLogger(__name__)

_registry: Dict[str, Any] = {}


def _model_path(name: str) -> Path:
    return MODELS_DIR / name


def save_lgbm(product: str, model) -> None:
    path = _model_path(f"lgbm_{product}.txt")
    model.booster_.save_model(str(path))


def load_lgbm(product: str):
    import lightgbm as lgb
    path = _model_path(f"lgbm_{product}.txt")
    if not path.exists():
        return None
    bst = lgb.Booster(model_file=str(path))
    return bst


def save_prophet_forecast(product: str, forecast_df) -> None:
    path = _model_path(f"prophet_fc_{product}.pkl")
    joblib.dump(forecast_df, path, compress=3)


def load_prophet_forecast(product: str):
    path = _model_path(f"prophet_fc_{product}.pkl")
    if not path.exists():
        return None
    return joblib.load(path)


def save_artifact(name: str, obj: Any) -> None:
    path = _model_path(f"{name}.pkl")
    joblib.dump(obj, path, compress=3)


def load_artifact(name: str) -> Optional[Any]:
    path = _model_path(f"{name}.pkl")
    if not path.exists():
        return None
    return joblib.load(path)


def models_ready() -> bool:
    """Check if minimum required models exist."""
    required = ["seasonal_data", "wfv_metrics", "feature_cols"]
    return all(_model_path(f"{r}.pkl").exists() for r in required)


def get_all_lgbm_feature_cols() -> Optional[Dict]:
    return load_artifact("feature_cols")