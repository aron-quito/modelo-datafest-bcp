import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import os

os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

print("==================================================")
print("🧬 LOCURA 2: Tabular Mixup (Clonación Mutante)")
print("==================================================")

df = pd.read_csv('../../dataset/train.csv')
X = df.drop(['id_cliente', 'objetivo'], axis=1)
y = df['objetivo']

X_num = X.select_dtypes(include=[np.number]).copy()
X_num.fillna(X_num.median(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=0.10, stratify=y, random_state=42
)

# PyTorch Simple MLP
class SimpleMLP(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

model = SimpleMLP(X_train_s.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCEWithLogitsLoss()

tensor_X = torch.tensor(X_train_s, dtype=torch.float32)
tensor_y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

print("🧟 Entrenando Red Neuronal con mutantes (Interpolación Matemática)...")
model.train()
batch_size = 1024
for epoch in range(40):
    permutation = torch.randperm(tensor_X.size()[0])
    for i in range(0, tensor_X.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = tensor_X[indices], tensor_y[indices]
        
        # Inyectar la locura Mixup
        mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.4)
        
        optimizer.zero_grad()
        outputs = model(mixed_x)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    test_outputs = torch.sigmoid(model(torch.tensor(X_test_s, dtype=torch.float32))).numpy()

auc = roc_auc_score(y_test, test_outputs)
print("\\n=======================================================")
print(f"⭐ AUC FINAL CON TABULAR MIXUP (Test 10%): {auc:.5f}")
print("=======================================================")
