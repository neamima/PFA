import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes=7):
    """
    Charge un modèle pré-entraîné et adapte sa dernière couche.
    """
    if model_name == 'resnet50':
        # Chargement de ResNet50 avec les poids ImageNet
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remplacement de la dernière couche (la tête)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'efficientnet_b0':
        # Modèle plus moderne, excellent ratio performance/poids
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'densenet121':
        # DenseNet est très utilisé en imagerie médicale
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError(f"Modèle {model_name} non reconnu.")
        
    return model