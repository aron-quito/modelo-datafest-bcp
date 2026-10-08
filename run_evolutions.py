import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
import shap
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.makedirs('locuras/22_client_aggregation_evolutions', exist_ok=True)

print("=== FASE 1: GENERANDO LA BASE AGREGADA (LA RECETA MÁGICA) ===")
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

# Foto Final
df_last = df.drop_duplicates(subset=['id_cliente'], keep='last').set_index('id_cliente')
y = df_last['objetivo']
df_last = df_last.drop(columns=['mes', 'objetivo'])

# Agregaciones (Media, Max, Min, Std)
aggs = {col: ['mean', 'max', 'min', 'std'] for col in num_cols}
df_grouped = df.groupby('id_cliente').agg(aggs)
df_grouped.columns = [f"{col}_{stat}" for col, stat in df_grouped.columns]

# Tendencias
df_first = df.drop_duplicates(subset=['id_cliente'], keep='first').set_index('id_cliente')
tendencias = pd.DataFrame(index=df_last.index)
for col in ['saldo_promedio', 'ratio_deuda_ingresos', 'ingresos']:
    tendencias[f'{col}_trend'] = df_last[col] - df_first[col]
tendencias['meses_en_sistema'] = df.groupby('id_cliente').size()

# Dataset Final
X = df_last.copy()
X = X.join(df_grouped)
X = X.join(tendencias)
X = X.fillna(0)
for col in bool_cols:
    X[col] = X[col].astype(int)

# 80/20 Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Base lista. Features: {X.shape[1]}. Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

resultados = {}

print("\n=== PIPELINE A: THE SURGEON (SHAP PRUNING) ===")
# Entrenamos un modelo rápido para sacar SHAP
model_base = CatBoostClassifier(iterations=300, depth=6, cat_features=cat_cols, verbose=0, random_seed=42)
model_base.fit(X_train, y_train)

explainer = shap.TreeExplainer(model_base)
shap_values = explainer.shap_values(X_train)
shap_sum = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({'feature': X_train.columns, 'shap_importance': shap_sum})
importance_df = importance_df.sort_values(by='shap_importance', ascending=False)
# Tiramos a la basura las 25 peores características
basura = importance_df.tail(25)['feature'].tolist()
print(f"Eliminando 25 features basura (ej. {basura[:3]}...)")

X_train_pruned = X_train.drop(columns=basura)
X_val_pruned = X_val.drop(columns=basura)
cat_cols_pruned = [c for c in cat_cols if c not in basura]

model_a = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6, cat_features=cat_cols_pruned, eval_metric='AUC', random_seed=42, verbose=0, early_stopping_rounds=50)
model_a.fit(X_train_pruned, y_train, eval_set=(X_val_pruned, y_val), use_best_model=True)
auc_a = roc_auc_score(y_val, model_a.predict_proba(X_val_pruned)[:, 1])
print(f"-> AUC Pipeline A (SHAP Pruning): {auc_a:.4f}")
resultados['Pipeline_A_SHAP'] = auc_a

print("\n=== PIPELINE B: THE BALANCER (SMOTE) ===")
# Para SMOTE necesitamos codificar numéricamente las categóricas temporalmente
te = TargetEncoder(cols=cat_cols)
X_train_num = te.fit_transform(X_train, y_train)
X_val_num = te.transform(X_val) # Validacion para evaluar luego

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_num, y_train)
print(f"SMOTE aplicado. Nueva distribución: 0: {sum(y_train_sm==0)}, 1: {sum(y_train_sm==1)}")

model_b = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6, eval_metric='AUC', random_seed=42, verbose=0, early_stopping_rounds=50)
model_b.fit(X_train_sm, y_train_sm, eval_set=(X_val_num, y_val), use_best_model=True)
auc_b = roc_auc_score(y_val, model_b.predict_proba(X_val_num)[:, 1])
print(f"-> AUC Pipeline B (SMOTE): {auc_b:.4f}")
resultados['Pipeline_B_SMOTE'] = auc_b

print("\n=== PIPELINE C: THE KAGGLE CHAMPION (LightGBM) ===")
# LightGBM prefiere que las categóricas sean 'category' o estén target-encoded
# Usaremos el Target-Encoded del paso anterior (X_train_num)
lgbm = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    colsample_bytree=0.7,
    subsample=0.8,
    random_state=42,
    class_weight='balanced'
)

# LightGBM requiere early stopping vía callbacks en nuevas versiones, pero por simplicidad
# confiaremos en un fit directo y mediremos (o usaremos un subset pequeño si hubiera callback, lo omitimos para compatibilidad)
lgbm.fit(X_train_num, y_train, eval_set=[(X_val_num, y_val)], eval_metric='auc', callbacks=[])
auc_c = roc_auc_score(y_val, lgbm.predict_proba(X_val_num)[:, 1])
print(f"-> AUC Pipeline C (LightGBM): {auc_c:.4f}")
resultados['Pipeline_C_LightGBM'] = auc_c

print("\n=== RESUMEN FINAL DE LAS EVOLUCIONES ===")
for pipe, score in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
    print(f"{pipe}: {score:.4f}")

with open('locuras/22_client_aggregation_evolutions/resultados.json', 'w') as f:
    json.dump(resultados, f, indent=4)
