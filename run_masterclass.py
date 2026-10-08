import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.cluster import MiniBatchKMeans
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.makedirs('locuras/15_masterclass_pipeline', exist_ok=True)

print("1. CARGANDO DATOS")
df = pd.read_csv('dataset/train.csv')

# Drop useless IDs
df = df.drop(columns=['id_cliente', 'mes'])

print("2. FEATURE ENGINEERING (Ingeniería Financiera)")
# Ratios lógicos que pueden separar el trigo de la paja
df['ingresos'] = df['ingresos'].replace(0, 1e-5)
df['ratio_saldo_ingreso'] = df['saldo_promedio'] / df['ingresos']
df['antiguedad_anos'] = df['antiguedad_cuenta_meses'] / 12.0
df['ratio_edad_cuenta'] = df['antiguedad_anos'] / df['edad']

cat_features = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
bool_features = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']

for col in bool_features:
    df[col] = df[col].astype(int)

# Separar X e y
X = df.drop(columns=['objetivo'])
y = df['objetivo']

# Split principal (Train / Val)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. TRANSFORMACIÓN NUMÉRICA TOTAL (Target Encoding)")
# Para poder usar K-Means y LigthGBM de forma numérica pura
te = TargetEncoder(cols=cat_features)
X_train_encoded = te.fit_transform(X_train, y_train)
X_val_encoded = te.transform(X_val)

print("4. AUDITORÍA DE VARIABLES (Permutation Importance aprox)")
# Entrenamos un LightGBM rápido
model_fs = lgb.LGBMClassifier(n_estimators=50, random_state=42, class_weight='balanced')
model_fs.fit(X_train_encoded, y_train)

# Extrayendo la importancia basada en GANANCIA de información (Gain)
importance = pd.DataFrame({
    'feature': X_train_encoded.columns,
    'gain': model_fs.booster_.feature_importance(importance_type='gain')
})
importance = importance.sort_values(by='gain', ascending=False)
print("Importancia de Variables (Top 10):")
print(importance.head(10))

# Eliminamos el 25% de las peores variables (puro ruido)
num_to_drop = int(len(importance) * 0.25)
useless_features = importance.tail(num_to_drop)['feature'].tolist()
print(f"Borrando {num_to_drop} variables inútiles: {useless_features}")

X_train_clean = X_train_encoded.drop(columns=useless_features)
X_val_clean = X_val_encoded.drop(columns=useless_features)

print("5. CLUSTERING-BASED UNDERSAMPLING (CBU)")
# Separamos las clases en el train limpio
X_train_0 = X_train_clean[y_train == 0]
X_train_1 = X_train_clean[y_train == 1]

# Clusterizar la clase mayoritaria (los "0"s) en 50 arquetipos
kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=1024)
clusters = kmeans.fit_predict(X_train_0)

# Añadimos temporalmente el cluster para poder hacer un muestreo estratificado
X_train_0_clustered = X_train_0.copy()
X_train_0_clustered['cluster'] = clusters

# Queremos reducir la clase 0 para que esté balanceada o con sobrepeso moderado
# Usaremos 1.5 veces más clase 0 que clase 1 para mantener un poco de la distribución original
target_n_0 = int(len(X_train_1) * 1.5)

# Tomamos una muestra aleatoria de los ceros, estratificada por 'cluster'
# Así garantizamos que el modelo vea ejemplos de TODAS las subpoblaciones de ceros
X_train_0_sampled = X_train_0_clustered.groupby('cluster', group_keys=False).apply(
    lambda x: x.sample(n=max(1, int(target_n_0 / 50)), random_state=42, replace=True) 
).drop(columns=['cluster'], errors='ignore')

# Reconstruimos el Dataset Balanceado
X_train_final = pd.concat([X_train_0_sampled, X_train_1])
y_train_final = np.array([0]*len(X_train_0_sampled) + [1]*len(X_train_1))

print(f"Dataset Final para entrenar: 0s={len(X_train_0_sampled)}, 1s={len(X_train_1)}")

print("6. ENTRENAMIENTO DEFINITIVO (LightGBM)")
final_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

final_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val_clean, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

print("\n--- EVALUACIÓN ---")
y_pred = final_model.predict(X_val_clean)
y_prob = final_model.predict_proba(X_val_clean)[:, 1]

auc = roc_auc_score(y_val, y_prob)
print(f"AUC: {auc:.4f}")
print("Classification Report:")
report = classification_report(y_val, y_pred)
print(report)

# Guardar Resultados
results = {
    'auc': float(auc),
    'useless_features_dropped': useless_features,
    'classification_report': classification_report(y_val, y_pred, output_dict=True)
}

with open('locuras/15_masterclass_pipeline/metricas.json', 'w') as f:
    json.dump(results, f, indent=4)

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Masterclass (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - LightGBM + CBU')
plt.legend()
plt.savefig('locuras/15_masterclass_pipeline/roc_curve.png')
plt.close()

# Feature Importance Final
fi = pd.DataFrame({'feature': X_train_final.columns, 'importance': final_model.booster_.feature_importance(importance_type='gain')})
fi = fi.sort_values('importance', ascending=False)
plt.figure(figsize=(10,8))
sns.barplot(x='importance', y='feature', data=fi)
plt.title('Importancia de Variables (Final)')
plt.tight_layout()
plt.savefig('locuras/15_masterclass_pipeline/feature_importance.png')
plt.close()

print("Proceso finalizado. Artefactos en locuras/15_masterclass_pipeline/")
