import os
import uuid
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.staticfiles import StaticFiles # <-- Ajoute ceci

# Import de tes modules personnalisés (situés dans src/)
from src.model import get_model
from src.db_auth import (
    authenticate_user, create_user, log_action, 
    save_diagnosis, get_user_diagnoses, get_db_connection,
    get_all_users, delete_user, get_all_logs
)

app = FastAPI(title="PeauIA API", version="2.0")
app.mount("/user_data", StaticFiles(directory="user_data"), name="user_data")

# 1. Configuration CORS pour React (port 5173 par défaut avec Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Chargement du Modèle PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_efficientnet_b0.pth"
CLASS_NAMES = {
    0: "Kératose bénigne (bkl)", 1: "Nævus / Grain de beauté (nv)",
    2: "Dermatofibrome (df)", 3: "Mélanome - MALIN (mel)",
    4: "Lésion vasculaire (vasc)", 5: "Carcinome basocellulaire (bcc)",
    6: "Kératose actinique (akiec)"
}

# Pipeline de transformation
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Chargement global du modèle au démarrage
model = get_model('efficientnet_b0', num_classes=len(CLASS_NAMES))
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✅ Modèle chargé sur {DEVICE}")
else:
    print("⚠️ Fichier de poids introuvable ! L'inférence ne fonctionnera pas.")

# 3. ROUTES AUTHENTIFICATION

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    success = create_user(username, password)
    if not success:
        raise HTTPException(status_code=400, detail="L'utilisateur existe déjà")
    return {"message": "Utilisateur créé avec succès"}

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Identifiants incorrects")
    
    log_action(user['id'], "Connexion via API")
    # On renvoie les infos de base (Dans une vraie app, on utiliserait un token JWT ici)
    return {
        "id": user['id'],
        "username": user['username'],
        "role": user['role']
    }

# 4. ROUTE PRÉDICTION (IA + BDD)

@app.post("/predict")
async def predict(
    user_id: int = Form(...),
    age: int = Form(...),
    sex: str = Form(...),
    localization: str = Form(...),
    file: UploadFile = File(...)
):
    # a. Sauvegarde locale de l'image
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
    upload_dir = "user_data"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # b. Inférence PyTorch
    try:
        image = Image.open(file_path).convert('RGB')
        img_tensor = transform_pipeline(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.max(probabilities, 0)
        
        prediction_label = CLASS_NAMES[top_idx.item()]
        prediction_score = float(top_prob.item()) * 100

        # c. Sauvegarde en BDD
        save_diagnosis(
            user_id, age, sex, localization, 
            file_path, prediction_label, prediction_score
        )
        
        return {
            "prediction": prediction_label,
            "probability": round(prediction_score, 2),
            "image_url": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse : {str(e)}")

# 5. ROUTES HISTORIQUE & ADMIN

@app.get("/history/{user_id}")
async def history(user_id: int):
    data = get_user_diagnoses(user_id)
    return data

@app.get("/admin/users")
async def admin_get_users():
    return get_all_users()

@app.delete("/admin/users/{user_id}")
async def admin_delete_user(user_id: int):
    try:
        delete_user(user_id)
        return {"message": "Utilisateur banni et supprimé avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/logs")
async def admin_get_logs():
    return get_all_logs()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)