import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report
import shap
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.makedirs('locuras/23_pipeline_a_validation', exist_ok=True)
os.makedirs('locuras/23_pipeline_a_validation/modelos', exist_ok=True)

print("1. GENERANDO LA BASE AGREGADA (Pipeline A Base)")
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

print(f"Base lista. Features: {X.shape[1]}. Clientes: {X.shape[0]}")

variantes = [
    {"name": "90_10_variante_1", "test_size": 0.1, "seed": 42},
    {"name": "90_10_variante_2", "test_size": 0.1, "seed": 777},
    {"name": "90_10_variante_3", "test_size": 0.1, "seed": 2026},
    {"name": "80_20_clasico", "test_size": 0.2, "seed": 42}
]

resultados = {}

for var in variantes:
    print(f"\n=== EJECUTANDO VARIANTE: {var['name']} (test_size={var['test_size']}, seed={var['seed']}) ===")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=var['test_size'], random_state=var['seed'], stratify=y)
    
    # 1. SHAP Pruning (Entrenado en Train para evitar leakage)
    model_base = CatBoostClassifier(iterations=300, depth=6, cat_features=cat_cols, verbose=0, random_seed=var['seed'])
    model_base.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model_base)
    shap_values = explainer.shap_values(X_train)
    shap_sum = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({'feature': X_train.columns, 'shap_importance': shap_sum})
    importance_df = importance_df.sort_values(by='shap_importance', ascending=False)
    
    # Eliminar las peores 25
    basura = importance_df.tail(25)['feature'].tolist()
    X_train_pruned = X_train.drop(columns=basura)
    X_val_pruned = X_val.drop(columns=basura)
    cat_cols_pruned = [c for c in cat_cols if c not in basura]

    # 2. Modelo Final
    model_final = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6, 
        cat_features=cat_cols_pruned, auto_class_weights='Balanced',
        eval_metric='AUC', random_seed=var['seed'], verbose=0, early_stopping_rounds=50
    )
    
    model_final.fit(X_train_pruned, y_train, eval_set=(X_val_pruned, y_val), use_best_model=True)
    
    # Evaluacion
    auc = roc_auc_score(y_val, model_final.predict_proba(X_val_pruned)[:, 1])
    print(f"AUC obtenido: {auc:.4f}")
    
    # Guardar modelo
    model_path = f"locuras/23_pipeline_a_validation/modelos/modelo_{var['name']}.cbm"
    model_final.save_model(model_path)
    
    resultados[var['name']] = {
        'auc': float(auc),
        'test_size': var['test_size'],
        'seed': var['seed'],
        'model_path': model_path,
        'basura_eliminada': basura
    }

print("\n=== RESUMEN FINAL ===")
for nombre, info in resultados.items():
    print(f"{nombre}: {info['auc']:.4f} AUC")

with open('locuras/23_pipeline_a_validation/resultados.json', 'w') as f:
    json.dump(resultados, f, indent=4)

print("\nProceso finalizado. Modelos y resultados guardados en locuras/23_pipeline_a_validation/")
