import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
import pandas as pd

# Import de tes modules personnalisés
from model import get_model
from dataset import get_dataloaders

# ==========================================================
# CONFIGURATION & HYPERPARAMÈTRES
# ==========================================================
# Chemins (à adapter selon ton arborescence)
TRAIN_PATH = "../data/raw/HAM10000_metadata"
TEST_PATH = "../data/processed/ISIC2018_Task3_Test_GroundTruth.csv"
IMG_DIR = "../data/images"
MODEL_SAVE_DIR = "../models"

# Paramètres d'exécution
# TEST_MODE = True pour ton PC portable, False pour le PC fixe (GPU)
TEST_MODE = True 
SAMPLE_SIZE = 100 if TEST_MODE else None 
EPOCHS = 2 if TEST_MODE else 15
BATCH_SIZE = 16 if TEST_MODE else 32
LEARNING_RATE = 0.001

# Liste des modèles à entraîner et comparer
MODELS_TO_COMPARE = ['resnet50', 'efficientnet_b0', 'densenet121']

# Détection automatique du matériel (GPU/MPS/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  Exécution sur : {device}")

# ==========================================================
# PRÉPARATION DES DONNÉES
# ==========================================================
train_loader, val_loader, class_to_idx, class_weights = get_dataloaders(
    TRAIN_PATH, TEST_PATH, IMG_DIR, batch_size=BATCH_SIZE, sample_size=SAMPLE_SIZE
)

class_weights = class_weights.to(device)    

# Inversion du dictionnaire pour l'affichage des noms de classes
idx_to_class = {v: k for k, v in class_to_idx.items()}
target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# ==========================================================
# BOUCLE DE COMPARAISON DES MODÈLES
# ==========================================================
results_summary = []

for model_name in MODELS_TO_COMPARE:
    print(f"\n{'='*60}")
    print(f"🚀 DÉBUT DE L'ENTRAÎNEMENT : {model_name.upper()}")
    print(f"{'='*60}")

    # Initialisation du modèle, perte et optimiseur
    model = get_model(model_name, num_classes=len(target_names)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- NOUVEAU : Application des poids à la fonction de perte ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- PHASE DE VALIDATION ---
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calcul des métriques
        current_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {train_loss/len(train_loader):.4f} - Val F1: {current_f1:.4f}")

        # Sauvegarde du meilleur état pour ce modèle précis
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            save_path = os.path.join(MODEL_SAVE_DIR, f"best_{model_name}.pth")
            torch.save(model.state_dict(), save_path)
            
    # --- FIN DE L'ENTRAÎNEMENT DU MODÈLE ---
    print(f"\n✅ Terminé pour {model_name}. Meilleur F1-Score : {best_val_f1:.4f}")
    
    # Affichage du rapport complet à la fin
    print(f"Rapport final pour {model_name} :")
    #print(classification_report(val_labels, val_preds, target_names=target_names, zero_division=0))
    print(classification_report(val_labels, val_preds, labels=range(len(target_names)), target_names=target_names, zero_division=0))

    results_summary.append({'model': model_name, 'best_f1': best_val_f1})

# ==========================================================
# RÉCAPITULATIF FINAL
# ==========================================================
print(f"\n{'='*60}")
print("🏆 RÉCAPITULATIF DE LA COMPARAISON")
print(f"{'='*60}")
for res in results_summary:
    print(f"- {res['model']} : F1-Score = {res['best_f1']:.4f}")
print(f"\nLes fichiers .pth sont disponibles dans le dossier '{MODEL_SAVE_DIR}'")