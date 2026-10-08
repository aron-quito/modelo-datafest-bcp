import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')

print("==================================================")
print("🔍 EVALUANDO EL MODELO EN LOS CEROS DESCARTADOS")
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

# SEPARAMOS CEROS Y UNOS DEL TRAIN
X_train_0 = X_train[y_train == 0].reset_index(drop=True)
y_train_0 = y_train[y_train == 0].reset_index(drop=True)

X_train_1 = X_train[y_train == 1].reset_index(drop=True)
y_train_1 = y_train[y_train == 1].reset_index(drop=True)

cantidad_unos = len(y_train_1)

# Mezclamos los Ceros aleatoriamente
from sklearn.utils import shuffle
X_train_0, y_train_0 = shuffle(X_train_0, y_train_0, random_state=42)

# LOS CEROS ELEGIDOS PARA ENTRENAR (14,910)
X_ceros_entrenamiento = X_train_0.iloc[:cantidad_unos]
y_ceros_entrenamiento = y_train_0.iloc[:cantidad_unos]

# LOS CEROS QUE TIRAMOS A LA BASURA EN LA PRUEBA 10 (aprox 69,000)
X_ceros_descartados = X_train_0.iloc[cantidad_unos:]
y_ceros_descartados = y_train_0.iloc[cantidad_unos:]

print(f"\\n2. Datos separados:")
print(f"   -> Ceros usados para entrenar: {len(X_ceros_entrenamiento)}")
print(f"   -> Unos usados para entrenar: {len(X_train_1)}")
print(f"   -> CEROS DESCARTADOS A EVALUAR: {len(X_ceros_descartados)}")

# Juntar para entrenamiento
X_train_final = pd.concat([X_ceros_entrenamiento, X_train_1])
y_train_final = pd.concat([y_ceros_entrenamiento, y_train_1])

train_pool = Pool(X_train_final, y_train_final, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

print("\\n3. Entrenando el modelo exactamente igual que en la Prueba 10...")
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    eval_metric='AUC',
    random_seed=42,
    verbose=False
)
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

print("\\n4. PONIENDO A PRUEBA EL MODELO CONTRA LOS CEROS DESCARTADOS...")
pool_descartados = Pool(X_ceros_descartados, cat_features=cat_cols)
preds_descartados = model.predict(pool_descartados)

correctos = sum(preds_descartados == 0)
incorrectos = sum(preds_descartados == 1)
precision_ceros = correctos / len(preds_descartados)

print("=======================================================")
print("🎯 RESULTADO EN LOS CEROS NUNCA ANTES VISTOS")
print("=======================================================")
print(f"Total de Ceros Descartados analizados: {len(preds_descartados)}")
print(f"✅ El modelo acertó y dijo que eran 'Ceros': {correctos} ({precision_ceros*100:.1f}%)")
print(f"❌ El modelo se equivocó y dijo que eran 'Unos': {incorrectos} ({(incorrectos/len(preds_descartados))*100:.1f}%)")
print("=======================================================")
print("Nota: No se puede calcular AUC porque el AUC requiere que haya al menos un '1' matemático en el set de prueba, y aquí el 100% son '0's.")
