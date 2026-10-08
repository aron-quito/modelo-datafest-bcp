import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import shap
import warnings
warnings.filterwarnings('ignore')

print("=== PRUEBA DE ROBUSTEZ: SPLIT 95/5 ===")

df = pd.read_csv('../dataset/train.csv')
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

# 95/5 Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42, stratify=y)
print(f"Split realizado. Train: {X_train.shape[0]} (95%), Val: {X_val.shape[0]} (5%)")

# Usamos la misma basura identificada por SHAP anteriormente (o podríamos recalcularla, pero para ser rápidos usamos la oficial)
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
X_train_pruned = X_train.drop(columns=basura)
X_val_pruned = X_val.drop(columns=basura)
cat_cols_pruned = [c for c in cat_cols if c not in basura]

model = CatBoostClassifier(
    iterations=1500, learning_rate=0.03, depth=6, 
    cat_features=cat_cols_pruned, auto_class_weights='Balanced',
    eval_metric='AUC', random_seed=42, verbose=100, early_stopping_rounds=50
)

model.fit(X_train_pruned, y_train, eval_set=(X_val_pruned, y_val), use_best_model=True)
auc = roc_auc_score(y_val, model.predict_proba(X_val_pruned)[:, 1])

print(f"\n======================================")
print(f"AUC FINAL en Split 95/5: {auc:.4f}")
print(f"======================================")
