import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time

# Import de l'architecture de ton modèle depuis le dossier src/
from src.model import get_model

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
st.title("🔬 Aide au Diagnostic Dermatologique par IA")
st.write("Uploadez une image dermoscopique pour obtenir une analyse algorithmique de la lésion.")

# Disclaimer Médical (Indispensable pour un PFA)
st.warning("**Avertissement :** Cette application est un prototype de recherche (PFA). Elle ne remplace en aucun cas l'avis d'un médecin spécialiste ou d'un dermatologue. Ne l'utilisez pas pour établir un diagnostic clinique définitif.")

# Zone d'upload
uploaded_file = st.file_uploader("Choisissez une image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Diviser l'écran en deux colonnes
    col1, col2 = st.columns(2)
    
    # Colonne de gauche : Affichage de l'image
    with col1:
        st.subheader("Image soumise")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        
    # Colonne de droite : Affichage des résultats
    with col2:
        st.subheader("Analyse de l'IA")
        
        with st.spinner("Analyse en cours par EfficientNet_B0..."):
            time.sleep(1) # Un petit délai simulé pour l'effet "réflexion de l'IA"
            model, device = load_model()
            top_probs, top_classes = predict_image(image, model, device)
            
            st.success("Analyse terminée !")
            
            # Affichage du Top 3 sous forme de jauges de progression
            st.write("**Top 3 des correspondances :**")
            
            for i in range(3):
                prob = top_probs[i]
                class_idx = top_classes[i]
                class_name = CLASS_NAMES[class_idx]
                
                # Mise en évidence visuelle si c'est un mélanome
                if "Mélanome" in class_name and prob > 10.0:
                    st.error(f"🚨 **{class_name} : {prob:.2f}%**")
                else:
                    st.write(f"{class_name} : {prob:.2f}%")
                    
                st.progress(int(prob) / 100.0)