import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
from sklearn.metrics import roc_auc_score, classification_report
import os

os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

print("==================================================")
print("🧠 RED NEURONAL RIGUROSA + SMOTETOMEK")
print("==================================================")

print("1. Preparando y Estandarizando Datos...")
df = pd.read_csv('../../dataset/train.csv')
X = df.drop(['id_cliente', 'objetivo'], axis=1)
y = df['objetivo']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y.values, test_size=0.10, stratify=y, random_state=42
)

cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
num_cols = [c for c in X.columns if c not in cat_cols]

# Imputar nulos simples
for col in num_cols:
    med = X_train_raw[col].median()
    X_train_raw[col] = X_train_raw[col].fillna(med)
    X_test_raw[col] = X_test_raw[col].fillna(med)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_train_s = preprocessor.fit_transform(X_train_raw)
X_test_s = preprocessor.transform(X_test_raw)

print("2. Aplicando SMOTETomek (Clonando Clase 1 y Borrando Gemelos)...")
smt = SMOTETomek(random_state=42, n_jobs=-1)
X_res, y_res = smt.fit_resample(X_train_s, y_train)

print(f"   Datos originales: {len(y_train)} -> Datos balanceados: {len(y_res)}")

# Red Neuronal con Arquitectura Robusta (Skip Connections / Residuals)
class RobustTabularNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.drop1(self.relu(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu(self.bn2(self.fc2(x))))
        x = self.relu(self.bn3(self.fc3(x)))
        return self.out(x)

model = RobustTabularNN(X_res.shape[1])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

tensor_X_train = torch.tensor(X_res, dtype=torch.float32)
tensor_y_train = torch.tensor(y_res, dtype=torch.float32).view(-1, 1)

# El test se evalúa sobre el mundo real (sin SMOTE)
tensor_X_test = torch.tensor(X_test_s, dtype=torch.float32)

print("3. Entrenando Arquitectura Rigurosa...")
batch_size = 512
model.train()

for epoch in range(20): # Pocas épocas porque hay el doble de datos por SMOTE
    permutation = torch.randperm(tensor_X_train.size()[0])
    for i in range(0, tensor_X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = tensor_X_train[indices], tensor_y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    test_outputs = torch.sigmoid(model(tensor_X_test)).numpy()
    preds = (test_outputs >= 0.5).astype(int)

auc = roc_auc_score(y_test, test_outputs)

print("\\n=======================================================")
print("🎯 REPORTE NN + SMOTETOMEK")
print("=======================================================")
print(classification_report(y_test, preds, digits=3))
print(f"⭐ AUC FINAL (NN Rigurosa): {auc:.5f}")
print("=======================================================")
