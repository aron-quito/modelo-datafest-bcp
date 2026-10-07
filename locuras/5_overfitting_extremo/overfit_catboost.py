import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

print("==================================================")
print("😈 OVERFITTING EXTREMO: CATBOOST")
print("==================================================")

df = pd.read_csv('../../dataset/train.csv')
X = df.drop(['id_cliente', 'objetivo'], axis=1)
y = df['objetivo']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

print("🌲 Entrenando CatBoost sin frenos (Profundidad 12, 3000 épocas, Cero Regularización)...")
train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

# Configuraciones malvadas para forzar el sobreajuste
model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.1,
    depth=12,             # Árboles ultra-profundos que memorizan a clientes individuales
    l2_leaf_reg=0.0,      # Apagamos la penalización matemática
    eval_metric='AUC',
    random_seed=42,
    verbose=500
)

# No usamos use_best_model=True, forzamos a que termine las 3000 iteraciones
model.fit(train_pool, eval_set=test_pool)

train_preds = model.predict_proba(train_pool)[:, 1]
test_preds = model.predict_proba(test_pool)[:, 1]

train_auc = roc_auc_score(y_train, train_preds)
test_auc = roc_auc_score(y_test, test_preds)

print("\\n=======================================================")
print(f"📈 AUC en ENTRENAMIENTO (Lo que el modelo cree saber): {train_auc:.5f}")
print(f"📉 AUC en TEST PURO (La dura realidad del mundo): {test_auc:.5f}")
print("=======================================================")
print(f"⚠️ Caída (Overfitting Gap): {train_auc - test_auc:.5f}")
