import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicTransformer
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')

print("==================================================")
print("🧬 LOCURA 4: Regresión Simbólica (Evolución Genética)")
print("==================================================")

df = pd.read_csv('../../dataset/train.csv')
X = df.drop(['id_cliente', 'objetivo'], axis=1)
y = df['objetivo']

X_num = X.select_dtypes(include=[np.number]).copy()
X_num.fillna(X_num.median(), inplace=True)

X_train_s, X_test_s, y_train, y_test, X_train_full, X_test_full = train_test_split(
    X_num, y, X, test_size=0.10, stratify=y, random_state=42
)

print("🧪 Evolucionando Ecuaciones Matemáticas con gplearn (tomará unos 30-60 seg)...")
# Usamos parámetros pequeños para que no tome 1 hora.
function_set = ['add', 'sub', 'mul', 'div', 'log', 'abs']
gp = SymbolicTransformer(generations=5, population_size=1000,
                         hall_of_fame=20, n_components=5,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=42, n_jobs=-1)

gp.fit(X_train_s, y_train)

print("\\n✅ Generación completada. Transformando datos con las 5 mejores ecuaciones alienígenas...")
gp_features_train = gp.transform(X_train_s)
gp_features_test = gp.transform(X_test_s)

for i in range(5):
    X_train_full[f'formula_alienigena_{i}'] = gp_features_train[:, i]
    X_test_full[f'formula_alienigena_{i}'] = gp_features_test[:, i]

cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
for col in cat_cols:
    X_train_full[col] = X_train_full[col].astype(str)
    X_test_full[col] = X_test_full[col].astype(str)

print("\\n🌲 Entrenando CatBoost con fórmulas genéticas...")
train_pool = Pool(X_train_full, y_train, cat_features=cat_cols)
test_pool = Pool(X_test_full, y_test, cat_features=cat_cols)

cb = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, eval_metric='AUC', verbose=0, random_seed=42)
cb.fit(train_pool, eval_set=test_pool, use_best_model=True)
best_auc = cb.get_best_score()['validation']['AUC']

print(f"\\n⭐ AUC FINAL CON REGRESIÓN SIMBÓLICA (Test 10%): {best_auc:.5f}")
