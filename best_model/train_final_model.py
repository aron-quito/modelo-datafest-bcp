import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, classification_report
import os

warnings.filterwarnings('ignore')

print("==================================================")
print("🏆 ENTRENANDO Y GUARDANDO EL MEJOR MODELO")
print("==================================================")

# 1. Cargar datos
print("Cargando dataset...")
df = pd.read_csv('../dataset/train.csv')
y = df['objetivo']

# 2. Generar Variables de Contexto Social (Tribu)
print("Generando Variables de Contexto Social (Tribu)...")
df_ctx = df.copy()
fin_cols = ['ingresos', 'saldo_promedio', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses']

for col in fin_cols:
    # Agrupamos por ocupación y región para hallar el promedio de la "Tribu"
    grupo_mean = df_ctx.groupby(['ocupacion', 'region'])[col].transform('mean')
    df_ctx[f'{col}_vs_tribu_ratio'] = df_ctx[col] / (grupo_mean + 1e-5)
    df_ctx[f'{col}_vs_tribu_diff'] = df_ctx[col] - grupo_mean

X = df_ctx.drop(['id_cliente', 'objetivo'], axis=1)

# 3. Preparar variables categóricas
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype(str)

# 4. Dividir para evaluación final
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

# 5. Entrenar Modelo Campeón
print("Entrenando CatBoost con Kaggle Trick (auto_class_weights='Balanced')...")
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    eval_metric='AUC',
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

# 6. Evaluación
preds = model.predict(test_pool)
proba = model.predict_proba(test_pool)[:, 1]
auc = roc_auc_score(y_test, proba)

print("\\n=======================================================")
print("🎯 REPORTE FINAL DEL MODELO")
print("=======================================================")
print(classification_report(y_test, preds, digits=3))
print(f"⭐ AUC FINAL: {auc:.5f}")

# 7. Guardar el Modelo
model_path = "campeon_catboost_tribu.cbm"
model.save_model(model_path)
print(f"\\n✅ Modelo guardado exitosamente como: {model_path}")
print("==================================================")
