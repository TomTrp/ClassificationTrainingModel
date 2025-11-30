from fastapi import FastAPI
from app.schemas import FruitFeatures
from app.utils import load_model
import pandas as pd

app = FastAPI(title="Fruit Classification API")

# Load model on startup
dt_model = load_model("fruit_dt_model.joblib")
rf_model = load_model("fruit_rf_model.joblib")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/dt")
def predict_dt(features: FruitFeatures):
    data = pd.DataFrame([{
        "size (cm)": features.size_cm,
        "weight (g)": features.weight_g,
        "avg_price (₹)": features.avg_price,
        "shape": features.shape,
        "color": features.color,
        "taste": features.taste
    }])
    prediction = dt_model.predict(data)[0]
    return {"model": "decision_tree", "prediction": prediction}

@app.post("/predict/rf")
def predict_rf(features: FruitFeatures):
    data = pd.DataFrame([{
        "size (cm)": features.size_cm,
        "weight (g)": features.weight_g,
        "avg_price (₹)": features.avg_price,
        "shape": features.shape,
        "color": features.color,
        "taste": features.taste
    }])
    prediction = rf_model.predict(data)[0]
    return {"model": "random_forest", "prediction": prediction}
