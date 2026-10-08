import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import MiniBatchKMeans
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.makedirs('locuras/17_cbu_catboost', exist_ok=True)

print("1. CARGANDO DATOS")
df = pd.read_csv('dataset/train.csv')

# Drop useless IDs
df = df.drop(columns=['id_cliente', 'mes'])

cat_features = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
bool_features = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']
num_features = [col for col in df.columns if col not in cat_features + bool_features + ['objetivo']]

for col in bool_features:
    df[col] = df[col].astype(int)

X = df.drop(columns=['objetivo'])
y = df['objetivo']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

print("2. CREANDO ESPACIO GEOMÉTRICO PARA CLUSTERING")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', TargetEncoder(), cat_features),
        ('bool', 'passthrough', bool_features)
    ])

# Creamos la versión numérica pura solo para que K-Means pueda "ver" distancias
X_train_geom = preprocessor.fit_transform(X_train, y_train)

print("3. CLUSTERING-BASED UNDERSAMPLING (CBU)")
# Clusterizamos TODO el train set (0s y 1s juntos) para mapear el terreno
n_clusters = 50
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048)
clusters = kmeans.fit_predict(X_train_geom)

# Analizamos qué tan "mezclado" está cada cluster
cluster_info = pd.DataFrame({'cluster': clusters, 'objetivo': y_train})
cluster_stats = cluster_info.groupby('cluster')['objetivo'].agg(['count', 'mean'])
cluster_stats.columns = ['total_clientes', 'porcentaje_unos']

indices_a_conservar = []

# Mantenemos todos los 1s siempre
indices_unos = cluster_info[cluster_info['objetivo'] == 1].index.tolist()
indices_a_conservar.extend(indices_unos)

ceros_eliminados_por_solapamiento = 0
ceros_eliminados_por_undersampling = 0

print("\n--- Estrategia CBU ---")
# Estrategia agresiva para despejar el camino a los 1s:
for c in range(n_clusters):
    stats = cluster_stats.loc[c]
    indices_ceros_aqui = cluster_info[(cluster_info['cluster'] == c) & (cluster_info['objetivo'] == 0)].index.tolist()
    
    if len(indices_ceros_aqui) == 0:
        continue
        
    porcentaje_1 = stats['porcentaje_unos']
    
    # REGLA 1: ZONA DE SOLAPAMIENTO SEVERO
    # Si el cluster tiene un alto porcentaje de 1s (ej. > 20%), es zona de los 1s.
    # ¡Borramos casi todos los 0s de aquí para limpiarles el territorio!
    if porcentaje_1 > 0.20:
        ceros_eliminados_por_solapamiento += len(indices_ceros_aqui)
        # No agregamos estos ceros a indices_a_conservar
        
    # REGLA 2: ZONA PURA (Casi solo 0s)
    # Hacemos undersampling clásico aleatorio para no abrumar al modelo
    else:
        # Nos quedamos con el 20% de los ceros de este cluster pacífico
        num_a_conservar = max(1, int(len(indices_ceros_aqui) * 0.20))
        ceros_elegidos = np.random.choice(indices_ceros_aqui, size=num_a_conservar, replace=False)
        indices_a_conservar.extend(ceros_elegidos)
        ceros_eliminados_por_undersampling += (len(indices_ceros_aqui) - num_a_conservar)

print(f"Ceros destruidos por invadir la zona de solapamiento: {ceros_eliminados_por_solapamiento}")
print(f"Ceros descartados pacíficamente en zonas puras: {ceros_eliminados_por_undersampling}")

print("\n4. EL PUENTE DE ÍNDICES: REGRESO AL MUNDO CRUDO")
# Usamos los índices para filtrar el dataset ORIGINAL (con los textos para CatBoost)
X_train_cbu = X_train.iloc[indices_a_conservar].copy()
y_train_cbu = y_train.iloc[indices_a_conservar].copy()

print(f"Distribución Final: {y_train_cbu.value_counts().to_dict()}")

print("\n5. ENTRENAMIENTO CATBOOST (Nativo)")
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
    X_train_cbu, y_train_cbu,
    eval_set=(X_val, y_val),
    use_best_model=True
)

print("\n6. EVALUACIÓN")
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
    'ceros_borrados_solapamiento': int(ceros_eliminados_por_solapamiento),
    'ceros_borrados_undersampling': int(ceros_eliminados_por_undersampling),
    'distribucion_train': y_train_cbu.value_counts().to_dict(),
    'classification_report': classification_report(y_val, y_pred, output_dict=True)
}

with open('locuras/17_cbu_catboost/metricas.json', 'w') as f:
    json.dump(results, f, indent=4)

fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'CBU + CatBoost (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Clustering-Based Undersampling')
plt.legend()
plt.savefig('locuras/17_cbu_catboost/roc_curve.png')
plt.close()

print("Proceso finalizado. Artefactos en locuras/17_cbu_catboost/")
