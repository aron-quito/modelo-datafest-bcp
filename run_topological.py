import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from imblearn.under_sampling import EditedNearestNeighbours
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier
import os
import json

# Crear carpeta de resultados
os.makedirs('locuras/14_topological_cleaning', exist_ok=True)

print("Cargando datos...")
df = pd.read_csv('dataset/train.csv')

# Separar X e y
X = df.drop(columns=['objetivo', 'id_cliente', 'mes'])
y = df['objetivo']

print(f"Distribución original: {y.value_counts().to_dict()}")

# Tipos de columnas
cat_features = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
bool_features = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']
num_features = [col for col in X.columns if col not in cat_features and col not in bool_features]

# Convertir booleanos a enteros temporales si es necesario, o dejarlos como están
for col in bool_features:
    X[col] = X[col].astype(int)

# 1. SPLIT (Crucial para no tener Data Leakage)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- FASE 1: Espacio Geométrico ---")
# Preprocesador para el ENN
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', TargetEncoder(), cat_features),
        ('bool', 'passthrough', bool_features)
    ])

# Transformamos solo el Train para el espacio de KNN
X_train_geom = preprocessor.fit_transform(X_train, y_train)

print("\n--- FASE 2: Limpieza Topológica (ENN) ---")
# ENN borrará los "0" que estén rodeados de "1"s
# Usamos n_neighbors=3 (por defecto). Kind_sel='all' (más conservador) o 'mode' (más agresivo)
enn = EditedNearestNeighbours(n_neighbors=3, kind_sel='all', n_jobs=-1)

# Fit y resample en el espacio geométrico. 
# Esto nos devuelve el dataset limpio PERO EN NÚMEROS
X_resampled_geom, y_resampled = enn.fit_resample(X_train_geom, y_train)

print(f"Distribución Post-Limpieza: {pd.Series(y_resampled).value_counts().to_dict()}")
ceros_borrados = (y_train == 0).sum() - (y_resampled == 0).sum()
print(f"ENN ha eliminado {ceros_borrados} clientes de clase '0' considerados RUIDO/SOLAPAMIENTO.")

print("\n--- FASE 3: Regreso al Espacio Crudo ---")
# ¡TRUCO MAGISTRAL! Rescatamos los índices de las filas que ENN decidió conservar
# imblearn guarda los indices en .sample_indices_ si el objeto soporte esta característica
# Como fit_resample devuelve los datos transformados, usamos sample_indices_
indices_retenidos = enn.sample_indices_

# Filtramos el TRAIN ORIGINAL CRUDO (con los textos intactos para CatBoost)
X_train_clean = X_train.iloc[indices_retenidos].copy()
y_train_clean = y_train.iloc[indices_retenidos].copy()

# Restaurar booleanos para CatBoost si se prefiere (CatBoost también lee ints, lo dejamos en int para evitar bugs)

print("\n--- FASE 4: Modelado CatBoost ---")
# Entrenamos con los datos RAW limpios. 
# Aún hay desbalance, así que auto_class_weights='Balanced'
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    auto_class_weights='Balanced',
    eval_metric='AUC',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

model.fit(
    X_train_clean, y_train_clean,
    eval_set=(X_val, y_val),
    use_best_model=True
)

print("\n--- EVALUACIÓN ---")
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
    'ceros_borrados_por_enn': int(ceros_borrados),
    'distribucion_original_train': y_train.value_counts().to_dict(),
    'distribucion_limpia_train': y_train_clean.value_counts().to_dict(),
    'classification_report': classification_report(y_val, y_pred, output_dict=True)
}

with open('locuras/14_topological_cleaning/metricas.json', 'w') as f:
    json.dump(results, f, indent=4)

# Matriz de confusion
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Limpieza Topológica')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.savefig('locuras/14_topological_cleaning/confusion_matrix.png')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Topological Cleaning (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend()
plt.savefig('locuras/14_topological_cleaning/roc_curve.png')
plt.close()

# Feature Importance
fi = pd.DataFrame({'feature': X_train_clean.columns, 'importance': model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
plt.figure(figsize=(10,8))
sns.barplot(x='importance', y='feature', data=fi)
plt.title('Importancia de Variables (Post-Limpieza Topológica)')
plt.tight_layout()
plt.savefig('locuras/14_topological_cleaning/feature_importance.png')
plt.close()

print("Proceso finalizado con éxito. Artefactos guardados en locuras/14_topological_cleaning/")
