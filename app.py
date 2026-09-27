import streamlit as st
import numpy as np
import joblib

st.title("Prédiction de la dépression")

st.caption("Projet pédagogique de machine learning : cet outil ne remplace en aucun cas un avis médical.")

model = joblib.load("models/modele_regression_logistique.pkl")
scaler = joblib.load("models/mon_scaler.pkl")

age               = st.slider("Âge", 0, 100, 20)
academic_pressure = st.slider("Pression académique (0 = aucune, 5 = extrême)", 0, 5, 2)
dietary_habits    = st.selectbox("Habitudes alimentaires", ["Healthy", "Moderate", "Unhealthy"])
suicidal_thoughts = st.selectbox("Pensées suicidaires ?", ["Non", "Oui"])
work_hours        = st.slider("Heures de travail/études par jour", 0, 12, 6)
financial_stress  = st.slider("Stress financier (0 = aucun, 5 = sévère)", 0, 5, 2)

dietary_map  = {"Healthy": 0, "Moderate": 1, "Unhealthy": 3}  # même encodage que LabelEncoder à l'entraînement
suicidal_map = {"Non": 0, "Oui": 1}

if st.button("Analyser"):
    X = np.array([[
        suicidal_map[suicidal_thoughts],  
        age,                           
        academic_pressure,          
        work_hours,                     
        financial_stress,              
        dietary_map[dietary_habits]       
    ]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = round(model.predict_proba(X_scaled)[0][1] * 100, 1)

    if pred == 1:
        st.error(f"Signes de dépression détectés — probabilité : **{prob}%**")
    else:
        st.success(f"Aucun signe majeur détecté — probabilité : **{prob}%**")

    st.info("Si tu traverses une période difficile, tu peux en parler à un professionnel de santé. "
            "En France, le 3114 (numéro national de prévention du suicide) est joignable 24h/24, gratuitement.")
