import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import os

os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

print("==================================================")
print("😈 OVERFITTING EXTREMO: RED NEURONAL (PyTorch)")
print("==================================================")

df = pd.read_csv('../../dataset/train.csv')
X = df.drop(['id_cliente', 'objetivo'], axis=1)
y = df['objetivo']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y.values, test_size=0.10, stratify=y, random_state=42
)

cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_train_s = preprocessor.fit_transform(X_train_raw)
X_test_s = preprocessor.transform(X_test_raw)

class OverfitMLP(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

model = OverfitMLP(X_train_s.shape[1])
# Optimizador agresivo sin Weight Decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCEWithLogitsLoss()

tensor_X = torch.tensor(X_train_s, dtype=torch.float32)
tensor_y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
tensor_X_test = torch.tensor(X_test_s, dtype=torch.float32)

print("🧠 Entrenando Cerebro Mediano (512 neuronas) para forzar Overfitting rápido...")
model.train()
batch_size = 512

for epoch in range(150):
    permutation = torch.randperm(tensor_X.size()[0])
    for i in range(0, tensor_X.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = tensor_X[indices], tensor_y[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            t_preds = torch.sigmoid(model(tensor_X)).numpy()
            v_preds = torch.sigmoid(model(tensor_X_test)).numpy()
            t_auc = roc_auc_score(y_train, t_preds)
            v_auc = roc_auc_score(y_test, v_preds)
            print(f"Época {epoch+1} -> Train AUC: {t_auc:.5f} | Test AUC: {v_auc:.5f}")
            
            if t_auc >= 0.70:
                print("🎯 ¡Objetivo alcanzado! Train AUC superó 0.70. Deteniendo el entrenamiento para probar...")
                break
        model.train()

model.eval()
with torch.no_grad():
    train_outputs = torch.sigmoid(model(tensor_X)).numpy()
    test_outputs = torch.sigmoid(model(tensor_X_test)).numpy()

final_train_auc = roc_auc_score(y_train, train_outputs)
final_test_auc = roc_auc_score(y_test, test_outputs)

print("\\n=======================================================")
print(f"📈 AUC en ENTRENAMIENTO (Ilusión): {final_train_auc:.5f}")
print(f"📉 AUC en TEST PURO (Realidad): {final_test_auc:.5f}")
print("=======================================================")
print(f"⚠️ Caída (Overfitting Gap): {final_train_auc - final_test_auc:.5f}")
