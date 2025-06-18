import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os
from utils.predict import predict_diabetes
from PIL import Image

# -------------------------
# CONFIGURATION GÉNÉRALE STREAMLIT
# -------------------------
st.set_page_config(
    page_title="Détection Diabète / Diabetes Detection",
    page_icon="🩺",
    layout="centered"
)

# Charger les variables d'environnement (.env)
USERNAME = st.secrets["APP_USERNAME"]
PASSWORD = st.secrets["APP_PASSWORD"]

# -------------------------
# LANGUE (sélecteur + dictionnaire)
# -------------------------
lang_options = ["Français", "English"]
selected_lang = st.sidebar.selectbox("🌐 Language / Langue", lang_options)
LANG = selected_lang  # devient une chaîne, pas un index


# Dictionnaire des traductions
translations = {
    "login_title": {
        "Français": "🔐 Authentification requise",
        "English": "🔐 Login Required"
    },
    "username_label": {
        "Français": "Nom d'utilisateur",
        "English": "Username"
    },
    "password_label": {
        "Français": "Mot de passe",
        "English": "Password"
    },
    "login_button": {
        "Français": "Se connecter",
        "English": "Log in"
    },
    "login_success": {
        "Français": "✅ Connexion réussie.",
        "English": "✅ Login successful."
    },
    "login_failed": {
        "Français": "❌ Identifiants incorrects.",
        "English": "❌ Incorrect credentials."
    }
}

# -------------------------
# AUTHENTIFICATION
# -------------------------
def authenticate():
    st.title(translations["login_title"][LANG])

    username_input = st.text_input(translations["username_label"][LANG])
    password_input = st.text_input(translations["password_label"][LANG], type="password")

    if st.button(translations["login_button"][LANG]):
        if username_input == USERNAME and password_input == PASSWORD:
            st.session_state["authenticated"] = True
            st.success(translations["login_success"][LANG])
        else:
            st.error(translations["login_failed"][LANG])


# Contrôle de session utilisateur
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    authenticate()
    st.stop()




# -------------------------
# TITRE / BANNIÈRE D'ACCUEIL
# -------------------------
if LANG == "Français":
    st.title("🩺 Application de Prédiction du Diabète")
else:
    st.title("🩺 Diabetes Prediction Application")

# -------------------------
# MENU LATÉRAL DE NAVIGATION
# -------------------------
menu_options = {
    "Français": [
        "🏠 Accueil",
        "🤖 Prédiction",
        "📈 Résultat",
        "💡 Recommandations",
        "📊 Explorations",
        "📤 Prédictions par CSV",
        "🆘 Aide / Contact"
    ],
    "English": [
        "🏠 Home",
        "🤖 Prediction",
        "📈 Result",
        "💡 Recommendations",
        "📊 Data Visualisation",
        "📤 Bulk Predictions (CSV)",
        "🆘 Help / Contact"
    ]
}

# Affichage du menu latéral
st.sidebar.title("🧭 Navigation")

# Affichage du logo dans la sidebar
logo_path = "assets\Diabetes_app_image.png"
try:
    logo = Image.open(logo_path)
    st.image(logo, use_container_width=True)

except:
    st.sidebar.warning("Logo introuvable.")

page = st.sidebar.radio(
    "Choisissez une page / Select a page", 
    menu_options[LANG]
)


# -------------------------
# ROUTAGE DE PAGES
# -------------------------

def show_home():
    # Chargement et affichage centré du logo
    logo_path = "assets\Diabetes_app_image.png"
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=180)
    except:
        st.warning("Logo non chargé.")

    if LANG == "Français":
        st.subheader("Bienvenue 👋")
        st.markdown("""
        Cette application a été conçue pour **prédire le risque de diabète** 
        à partir de données médicales simples.

        Elle vous permet de :
        - Faire une prédiction personnalisée
        - Visualiser les résultats
        - Obtenir des recommandations de santé
        - Importer un fichier CSV pour des prédictions en masse
        """)
    else:
        st.subheader("Welcome 👋")
        st.markdown("""
        This application is designed to **predict the risk of diabetes**
        from basic medical data.

        It allows you to:
        - Make an individual prediction
        - Visualize the results
        - Get health recommendations
        - Upload a CSV file for batch predictions
        """)

def show_prediction_form():
    if LANG == "Français":
        st.subheader("🧾 Formulaire de Prédiction")
        st.markdown("Veuillez remplir les informations médicales ci-dessous :")
    else:
        st.subheader("🧾 Prediction Form")
        st.markdown("Please fill in the medical information below:")

    # 🔧 Slider du seuil de décision AVANT tout
    threshold = st.slider(
        "🔧 Seuil de décision (0 = très sensible, 1 = très précis)" if LANG == "Français"
        else "🔧 Decision threshold (0 = sensitive, 1 = precise)",
        min_value=0.1, max_value=0.9, value=0.5, step=0.01
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Grossesses / Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glycémie (Glucose)", 0, 200, 100)
            blood_pressure = st.number_input("Pression artérielle", 0, 150, 70)
            skin_thickness = st.number_input("Épaisseur de peau", 0, 100, 20)

        with col2:
            insulin = st.number_input("Insuline", 0, 900, 80)
            bmi = st.number_input("IMC", 0.0, 70.0, 25.0, step=0.1)
            dpf = st.number_input("Antécédents familiaux", 0.0, 2.5, 0.5, step=0.01)
            age = st.number_input("Âge", 1, 120, 30)

        submit_button = st.form_submit_button("🔍 Prédire" if LANG == "Français" else "🔍 Predict")

    if submit_button:
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }

        prob, prediction = predict_diabetes(input_data, threshold=threshold)

        st.session_state["last_prediction"] = prediction
        st.session_state["last_proba"] = prob
        st.session_state["last_threshold"] = threshold

        st.success("✅ Prédiction effectuée !" if LANG == "Français" else "✅ Prediction completed!")
        st.info(f"📊 Probabilité estimée : {prob:.2%}")


def show_prediction_result():
    if "last_prediction" not in st.session_state or "last_proba" not in st.session_state:
        st.warning("⚠️ Aucune prédiction n'a encore été effectuée. Veuillez remplir le formulaire." if LANG == "Français"
                else "⚠️ No prediction made yet. Please fill out the form.")
        return

    prediction = st.session_state["last_prediction"]
    prob = st.session_state["last_proba"]
    threshold = st.session_state["last_threshold"]

    st.markdown("### 📈 Résultat de la prédiction")

    if LANG == "Français":
        st.markdown(f"**🔧 Seuil utilisé** : {threshold:.2f}")
        st.markdown(f"**📊 Probabilité estimée de diabète** : {prob:.2%}")
    else:
        st.markdown(f"**🔧 Threshold used**: {threshold:.2f}")
        st.markdown(f"**📊 Estimated diabetes probability**: {prob:.2%}")

    if prediction == 1:
        st.error("🚨 Le modèle indique que le patient est probablement **diabétique**." if LANG == "Français"
                 else "🚨 The model suggests the patient is likely **diabetic**.")
    else:
        st.success("✅ Le modèle indique que le patient est probablement **non diabétique**." if LANG == "Français"
                   else "✅ The model suggests the patient is likely **non-diabetic**.")


def show_recommendations():
    if "last_prediction" not in st.session_state:
        if LANG == "Français":
            st.warning("⚠️ Veuillez d'abord effectuer une prédiction pour afficher les recommandations.")
        else:
            st.warning("⚠️ Please make a prediction first to display recommendations.")
        return

    prediction = st.session_state["last_prediction"]

    st.subheader("💡 Recommandations" if LANG == "Français" else "💡 Recommendations")

    if prediction == 1:
        # Cas diabétique
        if LANG == "Français":
            st.error("🩺 Le patient présente un risque élevé de diabète.")
            st.markdown("""
            **Conseils pour la gestion du diabète :**
            - 🥗 Adoptez une alimentation équilibrée à faible indice glycémique
            - 🏃‍♂️ Faites de l'exercice régulièrement (30 min/jour)
            - 💧 Hydratez-vous correctement
            - 🚫 Réduisez les sucres rapides et les aliments transformés
            - 📅 Effectuez des contrôles réguliers chez un professionnel
            - 💊 Respectez les traitements médicaux si prescrits
            """)
        else:
            st.error("🩺 The patient shows a high risk of diabetes.")
            st.markdown("""
            **Tips for managing diabetes:**
            - 🥗 Follow a low-glycemic balanced diet
            - 🏃‍♂️ Exercise regularly (30 min/day)
            - 💧 Stay hydrated
            - 🚫 Avoid processed and sugary foods
            - 📅 Schedule regular check-ups
            - 💊 Follow medical prescriptions if any
            """)
    else:
        # Cas non diabétique
        if LANG == "Français":
            st.success("😊 Le patient semble en bonne santé.")
            st.markdown("""
            **Conseils pour conserver une bonne santé :**
            - 🥦 Mangez varié et évitez les excès de sucre
            - 🚶‍♀️ Marchez régulièrement
            - 📉 Surveillez votre poids et votre IMC
            - 🧘 Réduisez le stress
            - 🩺 Consultez votre médecin pour un suivi annuel
            """)
        else:
            st.success("😊 The patient appears to be healthy.")
            st.markdown("""
            **Tips to maintain good health:**
            - 🥦 Eat a varied diet and limit sugar
            - 🚶‍♀️ Walk regularly
            - 📉 Monitor your weight and BMI
            - 🧘 Reduce stress
            - 🩺 Visit your doctor for annual checkups
            """)


def show_data_viz():
    try:
        df = pd.read_csv("data/diabetes.csv")
    except FileNotFoundError:
        st.error("❌ Fichier 'diabetes.csv' introuvable dans le dossier 'data/'.")
        return

    if LANG == "Français":
        st.subheader("📊 Visualisation des Données")
        st.markdown("Explorez le jeu de données utilisé pour entraîner le modèle.")
    else:
        st.subheader("📊 Data Visualisation")
        st.markdown("Explore the dataset used to train the model.")

    # Distribution des classes
    st.markdown("### 📌 Distribution de la variable cible (Outcome)")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Outcome', ax=ax1, palette='pastel')
    ax1.set_xticklabels(['Non diabétique', 'Diabétique'] if LANG == "Français" else ['Non-diabetic', 'Diabetic'])
    ax1.set_title("Répartition des cas" if LANG == "Français" else "Case distribution")
    st.pyplot(fig1)

    # Heatmap de corrélation
    st.markdown("### 🔥 Corrélations entre les variables")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title("Matrice de corrélation" if LANG == "Français" else "Correlation matrix")
    st.pyplot(fig2)

    # Sélection de variable à explorer
    st.markdown("### 🔍 Exploration d'une variable")
    var = st.selectbox("Choisissez une variable" if LANG == "Français" else "Choose a variable", df.columns[:-1])

    fig3, ax3 = plt.subplots()
    sns.histplot(df[var], kde=True, ax=ax3, color="skyblue")
    ax3.set_title(f"Distribution de {var}" if LANG == "Français" else f"Distribution of {var}")
    st.pyplot(fig3)

    # Boxplot
    st.markdown("### 🧪 Boxplot par classe")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x='Outcome', y=var, data=df, palette='Set2', ax=ax4)
    ax4.set_xticklabels(['Non diabétique', 'Diabétique'] if LANG == "Français" else ['Non-diabetic', 'Diabetic'])
    ax4.set_title(f"{var} selon l’état de santé" if LANG == "Français" else f"{var} by health status")
    st.pyplot(fig4)


def show_bulk_prediction():
    st.subheader("📤 Prédiction en lot" if LANG == "Français" else "📤 Bulk Prediction via CSV")

    uploaded_file = st.file_uploader(
        "📁 Importez un fichier CSV avec les données des patients" if LANG == "Français"
        else "📁 Upload a CSV file with patient data",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}" if LANG == "Français" else f"Error reading file: {e}")
            return

        expected_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        if list(df.columns) != expected_columns:
            st.error("❌ Le fichier doit contenir les colonnes exactes suivantes :" if LANG == "Français"
                else "❌ The file must contain the following exact columns:")
            st.code(", ".join(expected_columns))
            return

        # Normalisation et prédiction
        from utils.predict import scaler, model
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)

        df['Prediction'] = predictions
        df['Prediction_Label'] = df['Prediction'].map({0: "Non diabétique" if LANG == "Français" else "Non-diabetic",
                                                    1: "Diabétique" if LANG == "Français" else "Diabetic"})

        st.success("✅ Prédictions générées !" if LANG == "Français" else "✅ Predictions generated!")
        st.dataframe(df)

        # Export CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Télécharger les résultats" if LANG == "Français" else "📥 Download Results",
            data=csv,
            file_name="predictions_result.csv",
            mime="text/csv"
        )
    else:
        st.info("📌 Veuillez importer un fichier pour commencer." if LANG == "Français"
                else "📌 Please upload a file to begin.")


def show_help():
    st.subheader("🆘 Aide / Contact" if LANG == "Français" else "🆘 Help / Contact")

    if LANG == "Français":
        st.markdown("""
        ### ℹ️ À propos de l'application
        Cette application utilise un modèle de Machine Learning (Gradient Boosting) pour prédire le **risque de diabète** à partir de données médicales simples.

        Elle n'a **pas vocation à remplacer un avis médical** et doit être utilisée à titre indicatif.

        ### 📧 Contact
        - Développeur : **Darryl MOMO**
        - Email : darrylmomo237@gmail.com
        - LinkedIn : [Voir le profil](https://www.linkedin.com/in/darryl-momo)
        - GitHub : [Accéder au dépôt](https://github.com/)

        ### 📝 Conseils d'utilisation
        - Utilisez des valeurs réalistes dans le formulaire
        - Exportez vos résultats pour en discuter avec un professionnel
        - Ne pas utiliser sur des données sensibles sans chiffrement
        """)
    else:
        st.markdown("""
        ### ℹ️ About this App
        This application uses a Gradient Boosting Machine Learning model to predict the **risk of diabetes** from simple medical data.

        It is **not intended to replace medical advice** and should be used for informational purposes only.

        ### 📧 Contact
        - Developer: **Darryl MOMO**
        - Email: darrylmomo237@gmail.com
        - LinkedIn: [View profile](https://www.linkedin.com/in/darryl-momo)
        - GitHub: [Access repository](https://github.com/)

        ### 📝 Usage Tips
        - Use realistic values in the prediction form
        - Export results to discuss with your doctor
        - Do not use on sensitive data without encryption
        """)


# -------------------------
# LOGIQUE D'AFFICHAGE DES PAGES
# -------------------------
if page == menu_options[LANG][0]:
    show_home()
elif page == menu_options[LANG][1]:
    show_prediction_form()
elif page == menu_options[LANG][2]:
    show_prediction_result()
elif page == menu_options[LANG][3]:
    show_recommendations()
elif page == menu_options[LANG][4]:
    show_data_viz()
elif page == menu_options[LANG][5]:
    show_bulk_prediction()
elif page == menu_options[LANG][6]:
    show_help()
