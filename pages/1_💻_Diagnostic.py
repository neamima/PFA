import streamlit as st
from src.db_auth import log_action
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time
import os
import uuid
from src.db_auth import save_diagnosis

# Import de l'architecture de ton modèle depuis le dossier src/
from src.model import get_model
# --- SÉCURITÉ : Vérification de la connexion ---
if not st.session_state.get('logged_in'):
    st.warning("Accès refusé. Veuillez vous connecter sur la page d'accueil.")
    st.stop() # Empêche le reste du code de s'exécuter
# -----------------------------------------------

# (Ici, colle tout le code de ton ancien app.py pour l'analyse d'image)
# ...

# ASTUCE POUR L'HISTORIQUE :
# Là où tu affiches le résultat ("st.success('Analyse terminée')"), ajoute cette ligne :
# log_action(st.session_state['user_id'], f"Analyse IA effectuée - Résultat Top 1 : {class_name}")



# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Détection de Lésions Cutanées (IA)",
    page_icon="🔬",
    layout="centered"
)

# ==========================================
# 2. DICTIONNAIRE DES CLASSES (Médicalisé)
# ==========================================
# On associe les acronymes du HAM10000 à des noms lisibles par un médecin/patient.
# IMPORTANT : L'ordre DOIT correspondre à l'ordre exact de tes matrices de confusion.
CLASS_NAMES = {
    0: "Kératose bénigne (bkl)",
    1: "Nævus mélanocytaire / Grain de beauté (nv)",
    2: "Dermatofibrome (df)",
    3: "Mélanome - MALIN (mel)",
    4: "Lésion vasculaire (vasc)",
    5: "Carcinome basocellulaire (bcc)",
    6: "Kératose actinique (akiec)"
}

# ==========================================
# 3. CHARGEMENT DU MODÈLE (En cache)
# ==========================================
# L'annotation @st.cache_resource évite de recharger le modèle à chaque clic de l'utilisateur
@st.cache_resource
def load_model():
    # On force l'utilisation du CPU pour le déploiement/démo sur le PC portable
    device = torch.device('cpu') 
    model = get_model('efficientnet_b0', num_classes=len(CLASS_NAMES))
    
    # Remplacer par le chemin exact vers ton fichier .pth
    model_path = "models/best_efficientnet_b0.pth" 
    
    # Chargement des poids (map_location='cpu' est vital si le modèle a été entraîné sur GPU)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# ==========================================
# 4. PIPELINE DE PRÉTRAITEMENT
# ==========================================
# Exactement le même que ton val_transforms !
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image, model, device):
    # Prétraitement de l'image
    img_tensor = transform_pipeline(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        # Transformation des sorties brutes en pourcentages (probabilités)
        probabilities = F.softmax(outputs, dim=1)[0] * 100
        
    # Trier les probabilités pour obtenir le Top 3
    top_prob, top_indices = torch.topk(probabilities, 3)
    
    return top_prob.numpy(), top_indices.numpy()

# ==========================================
# 5. INTERFACE UTILISATEUR (FRONT-END)
# ==========================================
# ... (Garde tout le haut du fichier app.py tel quel, jusqu'à la section 5) ...

# ==========================================
# 5. INTERFACE UTILISATEUR (FRONT-END)
# ==========================================
st.title("🔬 Aide au Diagnostic Dermatologique par IA")
st.write("Uploadez une image dermoscopique et renseignez le dossier patient.")

st.warning("**Avertissement :** Cette application est un prototype de recherche (PFA). Elle ne remplace en aucun cas l'avis d'un médecin.")

# --- NOUVEAU : DOSSIER PATIENT ---
st.markdown("### 📋 Informations Cliniques")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    age = st.number_input("Âge du patient", min_value=0, max_value=120, value=30, step=1)
with col_info2:
    sexe = st.selectbox("Sexe", ["Non spécifié", "Homme", "Femme"])
with col_info3:
    localisation = st.selectbox("Localisation de la lésion", 
                                ["Non spécifiée", "Dos", "Visage", "Tronc", "Bras/Jambe", "Cuir chevelu", "Main/Pied", "Autre"])

st.markdown("---")

# --- ZONE D'UPLOAD ---
st.markdown("### 📸 Imagerie")
uploaded_file = st.file_uploader("Choisissez une image dermoscopique (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image soumise")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        
        # Petit rappel pour le médecin
        st.info(f"**Patient :** {sexe}, {age} ans | **Zone :** {localisation}")
        
    with col2:
        st.subheader("Analyse de l'IA (Vision)")
        # Note pédagogique pour être transparent sur les capacités de l'IA
        st.caption("⚠️ *Note : L'algorithme actuel analyse uniquement les caractéristiques visuelles de la lésion, sans pondération par les données cliniques.*")
        
        with st.spinner("Analyse en cours par EfficientNet_B0..."):
            time.sleep(1)
            model, device = load_model()
            top_probs, top_classes = predict_image(image, model, device)
            
            st.success("Analyse terminée !")
            
            # --- NOUVEAU : SAUVEGARDE DU DIAGNOSTIC ---
            # 1. On crée un nom de fichier unique pour ne pas écraser d'anciennes images
            unique_filename = f"{uuid.uuid4().hex}.jpg"
            save_dir = "user_data"
            os.makedirs(save_dir, exist_ok=True) # Crée le dossier s'il n'existe pas
            image_path = os.path.join(save_dir, unique_filename)
            
            # 2. Sauvegarde physique de l'image
            image.save(image_path)
            
            # 3. Récupération du résultat gagnant (Top 1)
            top_1_class = CLASS_NAMES[top_classes[0]]
            top_1_prob = float(top_probs[0])
            
            # 4. Envoi à MySQL
            save_diagnosis(
                user_id=st.session_state['user_id'],
                age=age,
                sex=sexe,
                localization=localisation,
                image_path=image_path,
                top_prediction=top_1_class,
                top_probability=top_1_prob
            )
            # ------------------------------------------

            st.write("**Top 3 des correspondances visuelles :**")
            
            for i in range(3):
                prob = top_probs[i]
                class_idx = top_classes[i]
                class_name = CLASS_NAMES[class_idx]
                
                if "Mélanome" in class_name and prob > 10.0:
                    st.error(f"🚨 **{class_name} : {prob:.2f}%**")
                else:
                    st.write(f"{class_name} : {prob:.2f}%")
                    
                st.progress(int(prob) / 100.0)