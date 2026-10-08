import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("=== GENERADOR DE SUBMISSION PARA KAGGLE ===")

def preparar_datos(ruta_archivo, es_train=True):
    print(f"Cargando y procesando: {ruta_archivo}")
    df = pd.read_csv(ruta_archivo)
    df = df.sort_values(by=['id_cliente', 'mes'])

    num_cols = [
        'ingresos', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses',
        'numero_productos', 'saldo_promedio', 'dias_ultima_transaccion',
        'antiguedad_direccion_meses', 'visitas_web_ultimos_90_dias', 
        'distancia_sucursal_km', 'dia_preferido_pago', 'dias_ultima_interaccion', 'edad'
    ]
    bool_cols = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']
    
    df_last = df.drop_duplicates(subset=['id_cliente'], keep='last').set_index('id_cliente')
    
    y = None
    if es_train:
        y = df_last['objetivo']
        df_last = df_last.drop(columns=['mes', 'objetivo'])
    else:
        # Test no tiene objetivo, pero sí tiene mes, hay que quitarlo
        df_last = df_last.drop(columns=['mes'])

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
        
    return X, y

# 1. Preparar Train y Test
X_train, y_train = preparar_datos('../dataset/train.csv', es_train=True)
X_test, _ = preparar_datos('../dataset/test.csv', es_train=False)

# Asegurarse de que las columnas están en el mismo orden
X_test = X_test[X_train.columns]

# 2. Aplicar el SHAP Pruning (Eliminar las 25 basuras de ambos sets)
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
X_test_pruned = X_test.drop(columns=basura)

cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
cat_cols_pruned = [c for c in cat_cols if c not in basura]

# 3. Entrenar el Modelo Campeón en el 100% de la Data
print("Entrenando modelo con el 100% del Train Set...")
model_final = CatBoostClassifier(
    iterations=800, # Un promedio sano (en CV se detuvo entre 600 y 900)
    learning_rate=0.03, 
    depth=6, 
    cat_features=cat_cols_pruned, 
    auto_class_weights='Balanced',
    eval_metric='AUC', 
    random_seed=42, 
    verbose=100
)

# No usamos eval_set porque no hay validación, estamos usando el 100%
model_final.fit(X_train_pruned, y_train)

# 4. Predicciones en Test
print("Generando predicciones sobre el Test Set...")
probabilidades = model_final.predict_proba(X_test_pruned)[:, 1]

# 5. Formatear y Guardar
submission = pd.DataFrame({
    'id_cliente': X_test_pruned.index,
    'prediccion': probabilidades
})

submission.to_csv('submission_final_086.csv', index=False)
print("¡ÉXITO! Archivo 'submission_final_086.csv' creado con éxito.")
print(submission.head())
