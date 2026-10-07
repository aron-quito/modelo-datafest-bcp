import pandas as pd
import numpy as np
import warnings
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

print("==================================================")
print("🔬 ANÁLISIS DE SOLAPAMIENTO Y DESBALANCEO")
print("==================================================")

print("1. Cargando datos...")
df = pd.read_csv('../../dataset/train.csv')
X = df.drop(['id_cliente', 'objetivo'], axis=1)
y = df['objetivo']

print(f"\\n📊 DISTRIBUCIÓN DE CLASES:")
print(f"Clase 0 (No compró): {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
print(f"Clase 1 (Compró):    {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X_num = X[num_cols].fillna(X[num_cols].median())

# Standardize to make geometry distances (TomekLinks) fair
scaler = StandardScaler()
X_num_s = scaler.fit_transform(X_num)

print("\\n2. Calculando solapamiento en el hiperespacio (Tomek Links)...")
print("   Buscando clientes tipo '0' y tipo '1' que son estadísticamente gemelos.")
tl = TomekLinks(n_jobs=-1)
X_res, y_res = tl.fit_resample(X_num_s, y)

removed = len(y) - len(y_res)
print(f"⚠️ ¡Clientes solapados (Borrados por Tomek)!: {removed} de {len(y)} ({removed/len(y)*100:.2f}%)")
print(f"   Esto confirma qué tan fusionadas están las fronteras de decisión.")

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.10, stratify=y_res, random_state=42
)

print("\\n3. Entrenando Modelo sobre el Dataset Limpio de Solapamientos...")
model = CatBoostClassifier(iterations=300, depth=6, eval_metric='AUC', verbose=False, random_seed=42)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("\\n=======================================================")
print("🎯 REPORTE EXACTO DE CLASIFICACIÓN (Accuracy por Clase)")
print("=======================================================")
print(classification_report(y_test, preds, digits=3))

auc = roc_auc_score(y_test, proba)
print(f"⭐ AUC FINAL (Sin clientes solapados): {auc:.5f}")
print("=======================================================")
