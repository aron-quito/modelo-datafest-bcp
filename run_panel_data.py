import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.makedirs('locuras/20_panel_data', exist_ok=True)

print("1. CARGANDO DATOS (Modo Panel Data)")
df = pd.read_csv('dataset/train.csv')

print("2. FEATURE ENGINEERING: HISTORIAL DEL CLIENTE")
# Ordenamos por cliente y temporalidad
df = df.sort_values(by=['id_cliente', 'mes'])

# Creamos la variable sugerida: "Antigüedad / Historial en meses"
# Cuenta cuántas veces ha aparecido este cliente en el sistema hasta ese mes
df['meses_en_sistema'] = df.groupby('id_cliente').cumcount() + 1

# Variables a usar
cat_features = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
bool_features = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']

for col in bool_features:
    df[col] = df[col].astype(int)

# 3. EL SPLIT CORRECTO (GroupShuffleSplit)
# Nunca podemos separar las filas de un mismo cliente en Train y Val
# Todo el historial de un cliente debe ir completo a Train, o completo a Val.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df['id_cliente']))

train_data = df.iloc[train_idx].reset_index(drop=True)
val_data = df.iloc[val_idx].reset_index(drop=True)

X_train = train_data.drop(columns=['id_cliente', 'mes', 'objetivo'])
y_train = train_data['objetivo']

X_val = val_data.drop(columns=['id_cliente', 'mes', 'objetivo'])
y_val = val_data['objetivo']

print(f"Clientes únicos en Train: {train_data['id_cliente'].nunique()}")
print(f"Clientes únicos en Val:   {val_data['id_cliente'].nunique()}")

print("\n4. ENTRENAMIENTO CATBOOST (Panel Data Aware)")
model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    cat_features=cat_features,
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

print("\n5. EVALUACIÓN HONESTA (Sin Data Leakage)")
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_prob)
print(f"AUC: {auc:.4f}")
print("Classification Report:")
report = classification_report(y_val, y_pred)
print(report)

# Guardar Resultados
results = {
    'auc': float(auc),
    'feature_engineering': 'meses_en_sistema (cumcount)',
    'split_method': 'GroupShuffleSplit (no leakage)',
    'classification_report': classification_report(y_val, y_pred, output_dict=True)
}

with open('locuras/20_panel_data/metricas.json', 'w') as f:
    json.dump(results, f, indent=4)

# SHAP validation (Para ver si le gustó nuestra nueva variable)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
plt.figure(figsize=(10,8))
shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
plt.title('Importancia con Variable de Antiguedad')
plt.savefig('locuras/20_panel_data/shap_bar.png')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Panel Data Model (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Panel Data')
plt.legend()
plt.savefig('locuras/20_panel_data/roc_curve.png')
plt.close()

print("Proceso finalizado. Artefactos en locuras/20_panel_data/")
