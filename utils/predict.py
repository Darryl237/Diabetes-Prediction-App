import numpy as np
import joblib

# Chargement du modèle Gradient Boosting
model = joblib.load("models/gb_diabetes_model.pkl")

# Chargement du scaler
scaler = joblib.load("models/scaler.pkl")

def predict_diabetes(input_data: dict, threshold: float = 0.5) -> tuple:
    """
    Prédit le risque de diabète à partir des données utilisateur.

    Paramètres :
    -----------
    input_data : dict
        Contient les 8 variables médicales :
        - Pregnancies
        - Glucose
        - BloodPressure
        - SkinThickness
        - Insulin
        - BMI
        - DiabetesPedigreeFunction
        - Age

    Retour :
    --------
    int : 1 si diabétique, 0 sinon.
    """

    # Transformation du dictionnaire en array compatible
    input_array = np.array([list(input_data.values())])

    # Application du scaler (StandardScaler)
    input_scaled = scaler.transform(input_array)
    
    prob = model.predict_proba(input_scaled)[0][1]

    # Prédiction avec le modèle entraîné
    prediction = model.predict(input_scaled)[0]

    return prob, prediction
