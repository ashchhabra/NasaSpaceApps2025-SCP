import gradio as gr
from catboost import CatBoostClassifier
import joblib
import pickle
import numpy as np



# Load models properly
adaboost = CatBoostClassifier()
adaboost.load_model("models/adaboost_exoplanet.cbm")

random_forest = joblib.load("models/random_forest.pkl")
scaler = pickle.load(open("models/feature_scaler.pkl", "rb"))

# Feature names
features = ["planet_radii", "transit_depth", "days", "stars_radii", "earth_flux", "star_temp"]

# Prediction function
def predict_exoplanet(planet_radii, transit_depth, days, stars_radii, earth_flux, star_temp):
    X = np.array([[planet_radii, transit_depth, days, stars_radii, earth_flux, star_temp]])
    X_scaled = scaler.transform(X)

    pred1 = adaboost.predict(X_scaled)
    pred2 = random_forest.predict(X_scaled)

    # Ensemble voting (simple majority)
    final_pred = np.round((pred1 + pred2) / 2).astype(int)

    classes = ["false_positive", "candidate", "confirmed"]
    result = classes[int(final_pred[0])]

    return f"Predicted Class: **{result}**"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_exoplanet,
    inputs=[
        gr.Slider(0, 10, 1, label="Planet Radii"),
        gr.Slider(0, 1, 0.1, label="Transit Depth"),
        gr.Slider(0, 1000, 100, label="Days"),
        gr.Slider(0, 10, 1, label="Stars Radii"),
        gr.Slider(0, 1000, 100, label="Earth Flux"),
        gr.Slider(2000, 10000, 5000, label="Star Temp (K)")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Exoplanet Detection Model (AdaBoost + RandomForest)",
    description="This ensemble ML model predicts whether a celestial object is a confirmed exoplanet, a candidate, or a false positive.",
)

iface.launch()
