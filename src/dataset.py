import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = f"{self.dataframe.iloc[idx]['image_id']}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        label = self.dataframe.iloc[idx]['label']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224)) 

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(train_path, test_path, img_dir, batch_size=32, sample_size=None):
    """
    Charge le CSV, applique les transformations et retourne les DataLoaders.
    sample_size: Permet de prendre un petit échantillon pour tester sur le PC portable.
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    
    if sample_size:
        df_train = df_train.sample(n=sample_size, random_state=42).reset_index(drop=True)
        df_test = df_test.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Création des labels numériques
    classes = df_train['dx'].unique()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df_train['label'] = df_train['dx'].map(class_to_idx)
    df_test['label'] = df_test['dx'].map(class_to_idx)

    # Pour l'instant, on coupe bêtement le dataset en 80% train / 20% validation
    

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = HAM10000Dataset(df_train, img_dir, transform=train_transforms)
    val_dataset = HAM10000Dataset(df_test, img_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_to_idx