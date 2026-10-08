import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('locuras/21_client_aggregation', exist_ok=True)

print("1. CARGANDO DATOS PANEL")
df = pd.read_csv('dataset/train.csv')

# Ordenar cronológicamente para asegurar que 'last' y 'first' funcionen bien
df = df.sort_values(by=['id_cliente', 'mes'])

print("2. CREANDO AGREGACIONES FINANCIERAS Y ESTADÍSTICAS")

# Definir columnas
num_cols = [
    'ingresos', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses',
    'numero_productos', 'saldo_promedio', 'dias_ultima_transaccion',
    'antiguedad_direccion_meses', 'visitas_web_ultimos_90_dias', 
    'distancia_sucursal_km', 'dia_preferido_pago', 'dias_ultima_interaccion', 'edad'
]
cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
bool_cols = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']

# 2.1 Variables Estáticas o Último Estado
# Tomamos la última foto del cliente
df_last = df.drop_duplicates(subset=['id_cliente'], keep='last').set_index('id_cliente')
y = df_last['objetivo']
df_last = df_last.drop(columns=['mes', 'objetivo'])

# 2.2 Agregaciones Estadísticas sobre las variables numéricas
aggs = {}
for col in num_cols:
    aggs[col] = ['mean', 'max', 'min', 'std']

df_grouped = df.groupby('id_cliente').agg(aggs)
df_grouped.columns = [f"{col}_{stat}" for col, stat in df_grouped.columns]

# 2.3 Cálculo de Tendencias Financieras (Último mes - Primer mes)
df_first = df.drop_duplicates(subset=['id_cliente'], keep='first').set_index('id_cliente')

tendencias = pd.DataFrame(index=df_last.index)
for col in ['saldo_promedio', 'ratio_deuda_ingresos', 'ingresos']:
    tendencias[f'{col}_trend'] = df_last[col] - df_first[col]

# 2.4 Frecuencia de actividad
tendencias['meses_en_sistema'] = df.groupby('id_cliente').size()

# 2.5 Unir todo
X = df_last.copy() # Iniciamos con la última foto (incluye las categóricas actuales)
X = X.join(df_grouped)
X = X.join(tendencias)

# Llenar nulos creados por 'std' en clientes con solo 1 mes
X = X.fillna(0)

# Convertir booleanos a enteros
for col in bool_cols:
    X[col] = X[col].astype(int)

print(f"Dimensiones del dataset final (1 Fila por Cliente): {X.shape}")
print(f"Distribución del target: \n{y.value_counts()}")

print("3. ENTRENAMIENTO Y EVALUACIÓN (Validación 80/20)")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    cat_features=cat_cols,
    auto_class_weights='Balanced',
    eval_metric='AUC',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)

auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
y_pred = model.predict(X_val)
print(f"\n¡NUEVO AUC (Feature Aggregation): {auc:.4f}!")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Guardar la importancia de las variables
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({'feature': X_val.columns, 'shap_importance': shap_sum})
importance_df = importance_df.sort_values(by='shap_importance', ascending=False)
importance_df.to_csv('locuras/21_client_aggregation/shap_importances.csv', index=False)

print("Proceso finalizado.")
