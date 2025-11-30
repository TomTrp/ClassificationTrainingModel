from joblib import load
import pandas as pd

# Load model (Random Forest)
loaded_model = load("models/fruit_rf_model.joblib")

# Prepare data for test
new_data = pd.DataFrame({
    "size (cm)": [12.3],
    "weight (g)": [150],
    "avg_price (₹)": [40],
    "shape": ["long"],
    "color": ["yellow"],
    "taste": ["sweet"]
})

# Predict
prediction = loaded_model.predict(new_data)
print("Predicted fruit:", prediction[0])
