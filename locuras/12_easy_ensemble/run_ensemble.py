import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')

print("==================================================")
print("👥 EXPERIMENTO 12: ENSEMBLE DE UNDERSAMPLING")
print("==================================================")

df = pd.read_csv('../../dataset/train.csv')

# Variables de Tribu
print("1. Calculando Variables de Contexto Social (Tribu)...")
df_ctx = df.copy()
fin_cols = ['ingresos', 'saldo_promedio', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses']
for col in fin_cols:
    grupo_mean = df_ctx.groupby(['ocupacion', 'region'])[col].transform('mean')
    df_ctx[f'{col}_vs_tribu_ratio'] = df_ctx[col] / (grupo_mean + 1e-5)
    df_ctx[f'{col}_vs_tribu_diff'] = df_ctx[col] - grupo_mean

X = df_ctx.drop(['id_cliente', 'objetivo'], axis=1)
y = df_ctx['objetivo']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

print("2. Separando Ceros y Unos en Entrenamiento...")
X_train_0 = X_train[y_train == 0].reset_index(drop=True)
y_train_0 = y_train[y_train == 0].reset_index(drop=True)

X_train_1 = X_train[y_train == 1].reset_index(drop=True)
y_train_1 = y_train[y_train == 1].reset_index(drop=True)

# Vamos a crear N modelos para usar TODOS los ceros que antes borramos
num_ones = len(y_train_1)
num_zeros = len(y_train_0)
num_models = int(np.ceil(num_zeros / num_ones))  # Deberían ser unos 5 o 6 modelos

print(f"   Tenemos {num_zeros} Ceros y {num_ones} Unos.")
print(f"   Crearemos un Comité de {num_models} modelos para no desperdiciar a ningún Cero.")

X_train_0, y_train_0 = shuffle(X_train_0, y_train_0, random_state=42)

models = []
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

print("\\n3. Entrenando Comité de Expertos (Cada uno ve un pedazo distinto de Ceros)...")
for i in range(num_models):
    # Cortar un trozo de ceros
    start_idx = i * num_ones
    end_idx = min((i + 1) * num_ones, num_zeros)
    
    X_chunk_0 = X_train_0.iloc[start_idx:end_idx]
    y_chunk_0 = y_train_0.iloc[start_idx:end_idx]
    
    # Juntar con TODOS los unos
    X_batch = pd.concat([X_chunk_0, X_train_1], ignore_index=True)
    y_batch = pd.concat([y_chunk_0, y_train_1], ignore_index=True)
    
    train_pool = Pool(X_batch, y_batch, cat_features=cat_cols)
    
    print(f"   -> Entrenando Experto {i+1}/{num_models} (Tamaño: {len(y_batch)} clientes)...")
    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        eval_metric='AUC',
        random_seed=42 + i,
        verbose=False
    )
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=30)
    models.append(model)

print("\\n4. Evaluando el Promedio del Comité en el set de Pruebas (Test)...")
# Cada modelo vota su probabilidad, promediamos los votos
all_preds_proba = np.zeros(len(y_test))
for model in models:
    all_preds_proba += model.predict_proba(test_pool)[:, 1]

avg_proba = all_preds_proba / num_models
avg_preds = (avg_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, avg_proba)

print("\\n=======================================================")
print("🎯 REPORTE FINAL: COMITÉ DE UNDERSAMPLING")
print("=======================================================")
print(classification_report(y_test, avg_preds, digits=3))
print(f"⭐ AUC FINAL (Comité): {auc:.5f}")
print("=======================================================")
