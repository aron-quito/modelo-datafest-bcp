import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import warnings
warnings.filterwarnings('ignore')

os.makedirs('locuras/24_deep_learning_panel', exist_ok=True)

print("=== LOCURA 24: TABNET (ATTENTIVE TRANSFORMER) ===")
print("Usando la base agregada de la Locura 23 y aplicando Deep Learning.")

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

# 1. Poda de SHAP (Mantenemos la ventaja descubierta en la Locura 23)
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

# 2. Preprocesamiento para Redes Neuronales (TabNet necesita números)
# Codificamos las categóricas como enteros
cat_idxs = []
cat_dims = []
for col in cat_cols_pruned:
    le = LabelEncoder()
    X_pruned[col] = le.fit_transform(X_pruned[col].astype(str))
    cat_idxs.append(X_pruned.columns.get_loc(col))
    cat_dims.append(len(le.classes_))

# Split
X_train, X_val, y_train, y_val = train_test_split(X_pruned, y, test_size=0.1, random_state=42, stratify=y)

# Estandarización de las columnas continuas
con_cols = [c for c in X_pruned.columns if c not in cat_cols_pruned]
scaler = StandardScaler()
X_train[con_cols] = scaler.fit_transform(X_train[con_cols])
X_val[con_cols] = scaler.transform(X_val[con_cols])

# TabNet requiere matrices numpy
X_train_np = X_train.values
y_train_np = y_train.values
X_val_np = X_val.values
y_val_np = y_val.values

# 3. Entrenamiento de TabNet (Transformer Tabular)
print("Entrenando TabNet...")
clf = TabNetClassifier(
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=2,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":50, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax' # "entmax" es la atencion tipo transformer
)

# Pesos de clase manuales (TabNet no tiene auto_class_weights nativo)
weights = y_train.value_counts(normalize=True).to_dict()
# Invertimos los pesos
weights = {0: weights[1], 1: weights[0]}
class_weights = [weights[0] if y == 0 else weights[1] for y in y_train]

clf.fit(
    X_train=X_train_np, y_train=y_train_np,
    eval_set=[(X_val_np, y_val_np)],
    eval_name=['val'],
    eval_metric=['auc'],
    max_epochs=100, patience=20,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False
)

auc = roc_auc_score(y_val, clf.predict_proba(X_val_np)[:, 1])
print(f"\n======================================")
print(f"AUC FINAL TABNET (Locura 24): {auc:.4f}")
print(f"======================================")

with open('locuras/24_deep_learning_panel/metricas.txt', 'w') as f:
    f.write(f"AUC TabNet: {auc}\n")
