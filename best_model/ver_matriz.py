import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=== VERIFICANDO MATRIZ DE CONFUSIÓN (Pipeline Original) ===")
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

# Replicar el split 80/20 Clásico de la Variante 4 (seed=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Cargar el modelo guardado de esa variante
model = CatBoostClassifier()
model.load_model('modelos/modelo_80_20_clasico.cbm')

# Usar el orden exacto de variables con las que se entrenó el modelo
X_val_pruned = X_val[model.feature_names_]

from catboost import Pool
cat_cols_pruned = [c for c in model.feature_names_ if c in cat_cols]
pool_val = Pool(X_val_pruned, cat_features=cat_cols_pruned)
preds_class = model.predict(pool_val)

print("\n--- MATRIZ DE CONFUSIÓN ---")
cm = confusion_matrix(y_val, preds_class)
print("             Predicción: NO (0)   Predicción: SÍ (1)")
print(f"Real: NO (0)       {cm[0,0]}                  {cm[0,1]}")
print(f"Real: SÍ (1)       {cm[1,0]}                   {cm[1,1]}")

print("\n--- REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_val, preds_class))
