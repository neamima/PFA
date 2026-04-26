import streamlit as st
import os
from PIL import Image
from src.db_auth import get_user_diagnoses

st.set_page_config(page_title="Mon Historique", page_icon="📁", layout="wide")

# --- SÉCURITÉ : Vérification de la connexion ---
if not st.session_state.get('logged_in'):
    st.warning("Veuillez vous connecter pour voir votre historique.")
    st.stop()
# -----------------------------------------------

st.title("📁 Mon Historique de Diagnostics")
st.write(f"Dossier patient de : **{st.session_state['username']}**")
st.markdown("---")

# Récupération des données depuis MySQL
diagnoses = get_user_diagnoses(st.session_state['user_id'])

if not diagnoses:
    st.info("Vous n'avez pas encore effectué de diagnostic. Rendez-vous sur l'onglet 'Diagnostic' pour commencer.")
else:
    # Affichage sous forme de grille (3 colonnes) pour un beau design
    cols = st.columns(3)
    
    for index, diag in enumerate(diagnoses):
        # On alterne les colonnes pour chaque diagnostic (0, 1, 2, 0, 1, 2...)
        col = cols[index % 3]
        
        with col:
            st.markdown(f"### Diagnostic du {diag['created_at'].strftime('%d/%m/%Y')}")
            
            # Affichage de l'image
            if os.path.exists(diag['image_path']):
                img = Image.open(diag['image_path'])
                st.image(img, use_container_width=True)
            else:
                st.error("Image introuvable.")
                
            # Affichage des métadonnées
            st.write(f"**Patient :** {diag['age']} ans, {diag['sex']}")
            st.write(f"**Zone :** {diag['localization']}")
            
            # Mise en évidence si c'est un mélanome
            if "mel" in diag['top_prediction'] or "Mélanome" in diag['top_prediction']:
                st.error(f"🚨 **Prédiction :** {diag['top_prediction']} ({diag['top_probability']:.1f}%)")
            else:
                st.success(f"✅ **Prédiction :** {diag['top_prediction']} ({diag['top_probability']:.1f}%)")
            
            st.markdown("---")