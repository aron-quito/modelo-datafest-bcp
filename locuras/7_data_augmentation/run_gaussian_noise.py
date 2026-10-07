import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings('ignore')

print("==================================================")
print("🌀 DATA AUGMENTATION: CLONES CON RUIDO GAUSSIANO")
print("==================================================")

df = pd.read_csv('../../dataset/train.csv')
X = df.drop(['id_cliente', 'objetivo'], axis=1)
y = df['objetivo']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
num_cols = [c for c in X.columns if c not in cat_cols]

for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

print("1. Clonando la Clase 1 (Compradores)...")
# Separar solo la clase 1
X_train_ones = X_train[y_train == 1].copy()
y_train_ones = y_train[y_train == 1].copy()

print("2. Inyectando Ruido Gaussiano a los clones numéricos...")
# Por cada columna numérica, inyectamos un ruido muy sutil (2% de la desviación estándar)
np.random.seed(42)
for col in num_cols:
    std_dev = X_train_ones[col].std()
    noise = np.random.normal(loc=0.0, scale=0.02 * std_dev, size=len(X_train_ones))
    X_train_ones[col] = X_train_ones[col] + noise

# Unir los clones mutados al dataset de entrenamiento original
X_train_aug = pd.concat([X_train, X_train_ones], ignore_index=True)
y_train_aug = pd.concat([y_train, y_train_ones], ignore_index=True)

print(f"   Datos antes: {len(X_train)} -> Datos después del Ruido: {len(X_train_aug)}")

print("\\n3. Entrenando CatBoost con el Ejército de Clones Mutantes...")
train_pool = Pool(X_train_aug, y_train_aug, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

preds = model.predict(test_pool)
proba = model.predict_proba(test_pool)[:, 1]

auc = roc_auc_score(y_test, proba)

print("\\n=======================================================")
print("🎯 REPORTE NN + RUIDO GAUSSIANO")
print("=======================================================")
print(classification_report(y_test, preds, digits=3))
print(f"⭐ AUC FINAL (Clones Mutantes): {auc:.5f}")
print("=======================================================")
