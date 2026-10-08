import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

os.makedirs('locuras/25_stacking_ensemble', exist_ok=True)

print("=== LOCURA 25: HETEROGENEOUS STACKING (GOD LEVEL) ===")

df = pd.read_csv('dataset/train.csv')
df = df.sort_values(by=['id_cliente', 'mes'])

num_cols = [
    'ingresos', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses',
    'numero_productos', 'saldo_promedio', 'dias_ultima_transaccion',
    'antiguedad_direccion_meses', 'visitas_web_ultimos_90_dias', 
    'distancia_sucursal_km', 'dia_preferido_pago', 'dias_ultima_interaccion', 'edad'
]
cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
bool_cols = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']

df_last = df.drop_duplicates(subset=['id_cliente'], keep='last').set_index('id_cliente')
y = df_last['objetivo']
df_last = df_last.drop(columns=['mes', 'objetivo'])

aggs = {col: ['mean', 'max', 'min', 'std'] for col in num_cols}
df_grouped = df.groupby('id_cliente').agg(aggs)
df_grouped.columns = [f"{col}_{stat}" for col, stat in df_grouped.columns]

df_first = df.drop_duplicates(subset=['id_cliente'], keep='first').set_index('id_cliente')
tendencias = pd.DataFrame(index=df_last.index)
for col in ['saldo_promedio', 'ratio_deuda_ingresos', 'ingresos']:
    tendencias[f'{col}_trend'] = df_last[col] - df_first[col]
tendencias['meses_en_sistema'] = df.groupby('id_cliente').size()

X = df_last.copy()
X = X.join(df_grouped)
X = X.join(tendencias)
X = X.fillna(0)
for col in bool_cols:
    X[col] = X[col].astype(int)

# Poda de SHAP (49 variables top)
basura = [
    "dia_preferido_pago_min", "dia_preferido_pago_max", "distancia_sucursal_km",
    "antiguedad_direccion_meses_mean", "dia_preferido_pago", "visitas_web_ultimos_90_dias_max",
    "visitas_web_ultimos_90_dias", "dia_preferido_pago_mean", "visitas_web_ultimos_90_dias_mean",
    "es_nuevo_cliente", "tiene_seguro", "ingresos_trend", "ratio_deuda_ingresos_trend",
    "saldo_promedio_trend", "edad_std", "dia_preferido_pago_std", "ingresos_std",
    "ratio_deuda_ingresos_std", "antiguedad_direccion_meses_std", "distancia_sucursal_km_std",
    "antiguedad_cuenta_meses_std", "saldo_promedio_std", "dias_ultima_transaccion_std",
    "visitas_web_ultimos_90_dias_std", "numero_productos_std"
]
X_pruned = X.drop(columns=basura)
cat_cols_pruned = [c for c in cat_cols if c not in basura]

# Preparar variables para que Scikit-Learn no se rompa (One-Hot Encoding)
X_pruned = pd.get_dummies(X_pruned, columns=cat_cols_pruned, drop_first=True)

# Split
X_train, X_val, y_train, y_val = train_test_split(X_pruned, y, test_size=0.1, random_state=42, stratify=y)

print("Iniciando entrenamiento del Stacking Ensemble...")
print("Meta-Modelo aprenderá a combinar CatBoost, LightGBM y XGBoost.")

# 1. Base Models (sin parámetros problemáticos de categorías)
estimators = [
    ('cat', CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        auto_class_weights='Balanced', verbose=0, random_seed=42
    )),
    ('lgb', LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        class_weight='balanced', random_state=42, verbose=-1
    )),
    ('xgb', XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
        random_state=42, verbosity=0
    ))
]

# 2. Meta-Model
meta_model = LogisticRegression()

# 3. Stacking
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=3,
    stack_method='predict_proba',
    n_jobs=1,
    verbose=1
)

stack.fit(X_train, y_train)

# Evaluacion
probs = stack.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, probs)

print(f"\n======================================")
print(f"AUC FINAL STACKING ENSEMBLE: {auc:.4f}")
print(f"======================================")

# Mostrar pesos que el meta-modelo asignó a cada algoritmo
print("\nPesos del Meta-Modelo (Importancia de cada algoritmo):")
print(f"CatBoost: {stack.final_estimator_.coef_[0][0]:.4f}")
print(f"LightGBM: {stack.final_estimator_.coef_[0][1]:.4f}")
print(f"XGBoost:  {stack.final_estimator_.coef_[0][2]:.4f}")
