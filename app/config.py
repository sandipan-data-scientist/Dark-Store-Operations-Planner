from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "delhi_ncr_darkstore_fruit_vegetable_sales_2022_2025.csv"

API_HOST = os.getenv("API_HOST", "http://localhost")
API_PORT = int(os.getenv("PORT_API", 8000))
API_BASE_URL = f"{API_HOST}:{API_PORT}"

FRUITS = [
    "apple_kg", "banana_kg", "guava_kg", "mangoes_kg", "pomegranate_kg",
    "orange_kg", "lemon_kg", "grapes_kg", "watermelon_kg", "muskmelon_kg",
]
VEGETABLES = [
    "onion_kg", "carrot_kg", "garlic_kg", "coriander_kg", "cucumber_kg",
    "radish_kg", "cabbage_kg", "cauliflower_kg", "tomato_kg", "potato_kg",
    "ginger_kg", "broccoli_kg", "lady_finger_kg", "chillies_kg", "brinjal_kg",
    "bitter_gourd_kg", "spinach_kg", "pumpkin_kg", "pointed_gourd_kg",
    "beetroot_kg", "capsicum_kg", "green_peas_kg",
]
ALL_PRODUCTS = FRUITS + VEGETABLES

PRETTY_NAMES = {col: col.replace("_kg", "").replace("_", " ").title() for col in ALL_PRODUCTS}
REVERSE_NAMES = {v: k for k, v in PRETTY_NAMES.items()}

# Typical shelf-life defaults (days) for perishables
DEFAULT_SHELF_LIFE = {
    "apple_kg": 14, "banana_kg": 5, "guava_kg": 4, "mangoes_kg": 5,
    "pomegranate_kg": 14, "orange_kg": 14, "lemon_kg": 14, "grapes_kg": 5,
    "watermelon_kg": 7, "muskmelon_kg": 5,
    "onion_kg": 30, "carrot_kg": 14, "garlic_kg": 90, "coriander_kg": 3,
    "cucumber_kg": 7, "radish_kg": 5, "cabbage_kg": 7, "cauliflower_kg": 7,
    "tomato_kg": 5, "potato_kg": 21, "ginger_kg": 21, "broccoli_kg": 5,
    "lady_finger_kg": 3, "chillies_kg": 7, "brinjal_kg": 5,
    "bitter_gourd_kg": 5, "spinach_kg": 2, "pumpkin_kg": 21,
    "pointed_gourd_kg": 5, "beetroot_kg": 14, "capsicum_kg": 7,
    "green_peas_kg": 3,
}

PROPHET_PARAMS = {
    "seasonality_mode": "multiplicative",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": 0.95,
}

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

FORECAST_HORIZON_DAYS = 365
LAG_FEATURES = (7, 14, 30, 60, 365)
ROLLING_WINDOWS = (7, 14, 30, 90)