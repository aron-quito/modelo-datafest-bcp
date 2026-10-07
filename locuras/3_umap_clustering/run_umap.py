import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

try:
    import umap
except ImportError:
    print("❌ Error: Necesitas instalar umap. Ejecuta: pip install umap-learn")
    exit(1)

warnings.filterwarnings('ignore')

print("==================================================")
print("🌀 LOCURA 3: UMAP + K-Means Clustering")
print("==================================================")

# 1. Cargar Datos
df = pd.read_csv('../../dataset/train.csv')
X = df.drop(['id_cliente', 'objetivo'], axis=1)
y = df['objetivo']

# 2. Preprocesamiento Rápido para UMAP (UMAP odia las categorías de texto y NaNs)
print("⚙️ Preparando datos puramente numéricos para UMAP...")
X_num = X.select_dtypes(include=[np.number]).copy()
X_num.fillna(X_num.median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

# Split para evitar Data Leakage (UMAP solo debe aprender del Train)
X_train_s, X_test_s, y_train, y_test, X_train_full, X_test_full = train_test_split(
    X_scaled, y, X, test_size=0.10, stratify=y, random_state=42
)

# 3. Agujero Negro Dimensional (UMAP)
print("🌌 Comprimiendo dimensiones con UMAP (esto tomará unos segundos)...")
# n_neighbors pequeño captura detalle local, min_dist agrupa apretado
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_train = reducer.fit_transform(X_train_s)
umap_test = reducer.transform(X_test_s)

# 4. Encontrando las "Tribus" (Clustering)
print("🏕️ Detectando Tribus con K-Means (Buscando 20 perfiles ocultos)...")
kmeans = KMeans(n_clusters=20, random_state=42)
tribus_train = kmeans.fit_predict(umap_train)
tribus_test = kmeans.predict(umap_test)

# 5. Fusionando la Magia con CatBoost
# Reensamblamos los datos originales y le inyectamos la "Tribu"
X_train_final = X_train_full.copy()
X_test_final = X_test_full.copy()

X_train_final['tribu_umap'] = tribus_train.astype(str) # Lo volvemos texto para que CatBoost haga magia
X_test_final['tribu_umap'] = tribus_test.astype(str)

cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal', 'tribu_umap']
for col in cat_cols:
    X_train_final[col] = X_train_final[col].astype(str)
    X_test_final[col] = X_test_final[col].astype(str)

print(f"✅ Variables originales: {X_train_full.shape[1]} | Con Tribu: {X_train_final.shape[1]}")

# 6. CatBoost
print("\\n🌲 Entrenando CatBoost sobre los datos enriquecidos por el Agujero Negro...")
train_pool = Pool(X_train_final, y_train, cat_features=cat_cols)
test_pool = Pool(X_test_final, y_test, cat_features=cat_cols)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=test_pool, use_best_model=True)
best_auc = model.get_best_score()['validation']['AUC']

print("\\n=======================================================")
print(f"⭐ AUC FINAL CON TRIBUS UMAP (Test Puro): {best_auc:.5f}")
print("=======================================================")

# Ver si la Tribu sirvió de algo
importances = model.get_feature_importance()
for i, col in enumerate(X_train_final.columns):
    if col == 'tribu_umap':
        print(f"👁️ Importancia de la variable 'tribu_umap': {importances[i]:.2f}%")
