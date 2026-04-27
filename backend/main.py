from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Initialisation de l'API
app = FastAPI(title="API IA Mélanome", version="1.0")

# Configuration CORS (CRUCIAL : Autorise ton Front React à parler à ton Back Python)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Port par défaut de Vite/React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "L'API PyTorch est en ligne ! 🚀"}

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu_available": False} # On ajustera selon ton PC