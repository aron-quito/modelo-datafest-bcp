import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, Pool
import torch
import torch.nn as nn
import os

os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

print("==================================================")
print("🎭 LOCURA 1: Self-Supervised Masking (Autoencoder)")
print("==================================================")

df = pd.read_csv('../../dataset/train.csv')
X = df.drop(['id_cliente', 'objetivo'], axis=1)
y = df['objetivo']

# Solo usaremos las numéricas para simplificar la reconstrucción matemática
X_num = X.select_dtypes(include=[np.number]).copy()
X_num.fillna(X_num.median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

X_train_s, X_test_s, y_train, y_test, X_train_full, X_test_full = train_test_split(
    X_scaled, y, X, test_size=0.10, stratify=y, random_state=42
)

# PyTorch Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self, num_features, latent_dim=12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.3), # MASKING: Borra el 30% de las variables aleatoriamente
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_features)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

print("🧠 Entrenando la Red para adivinar datos faltantes (Self-Supervised)...")
model = DenoisingAutoencoder(num_features=X_train_s.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

tensor_X = torch.tensor(X_train_s, dtype=torch.float32)

# Entrenar por 50 épocas
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    decoded, _ = model(tensor_X)
    loss = criterion(decoded, tensor_X) # Intentar reconstruir el original pese al 30% borrado
    loss.backward()
    optimizer.step()

print("✅ Física del banco aprendida. Extrayendo Latent Space...")
model.eval()
with torch.no_grad():
    _, train_latent = model(tensor_X)
    _, test_latent = model(torch.tensor(X_test_s, dtype=torch.float32))

# Agregar los 12 vectores latentes como nuevas variables
for i in range(12):
    X_train_full[f'latent_{i}'] = train_latent[:, i].numpy()
    X_test_full[f'latent_{i}'] = test_latent[:, i].numpy()

cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
for col in cat_cols:
    X_train_full[col] = X_train_full[col].astype(str)
    X_test_full[col] = X_test_full[col].astype(str)

print("\\n🌲 Entrenando CatBoost con Representaciones Latentes...")
train_pool = Pool(X_train_full, y_train, cat_features=cat_cols)
test_pool = Pool(X_test_full, y_test, cat_features=cat_cols)

cb = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, eval_metric='AUC', verbose=0, random_seed=42)
cb.fit(train_pool, eval_set=test_pool, use_best_model=True)
best_auc = cb.get_best_score()['validation']['AUC']

print(f"\\n⭐ AUC FINAL CON MASKING LATENTE (Test 10%): {best_auc:.5f}")
