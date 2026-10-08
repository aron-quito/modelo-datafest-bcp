import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings('ignore')

print("==================================================")
print("✂️ EXPERIMENTO 10: UNDERSAMPLING EXTREMO")
print("==================================================")

df = pd.read_csv('../../dataset/train.csv')

# Usaremos las variables de Tribu (nuestro mejor set)
print("1. Calculando Variables de Contexto Social (Tribu)...")
df_ctx = df.copy()
fin_cols = ['ingresos', 'saldo_promedio', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses']
for col in fin_cols:
    grupo_mean = df_ctx.groupby(['ocupacion', 'region'])[col].transform('mean')
    df_ctx[f'{col}_vs_tribu_ratio'] = df_ctx[col] / (grupo_mean + 1e-5)
    df_ctx[f'{col}_vs_tribu_diff'] = df_ctx[col] - grupo_mean

X = df_ctx.drop(['id_cliente', 'objetivo'], axis=1)
y = df_ctx['objetivo']

# Convertir categóricas a string
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype(str)

print(f"2. Distribución Original: {len(y[y==0])} Ceros vs {len(y[y==1])} Unos")

# Primero dividimos en train/test (El Test NUNCA se toca para que sea real)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

# Aplicar Undersampling SOLO al entrenamiento
print("3. Aplicando Undersampling (Borrando Ceros masivamente)...")
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

print(f"   Distribución Entrenamiento después de Undersampling:")
print(f"   Ceros: {len(y_train_res[y_train_res==0])} | Unos: {len(y_train_res[y_train_res==1])}")

train_pool = Pool(X_train_res, y_train_res, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

print("\\n4. Entrenando CatBoost con el dataset reducido a la mitad...")
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    eval_metric='AUC',
    # IMPORTANTE: NO usamos pesos aquí, porque ya balanceamos físicamente los datos
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

preds = model.predict(test_pool)
proba = model.predict_proba(test_pool)[:, 1]
auc = roc_auc_score(y_test, proba)

print("\\n=======================================================")
print("🎯 REPORTE FINAL: UNDERSAMPLING")
print("=======================================================")
print(classification_report(y_test, preds, digits=3))
print(f"⭐ AUC FINAL (Undersampling): {auc:.5f}")
print("=======================================================")
