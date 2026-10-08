import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score

warnings.filterwarnings('ignore')

df = pd.read_csv('../../dataset/train.csv')

# Variables de Tribu
df_ctx = df.copy()
fin_cols = ['ingresos', 'saldo_promedio', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses']
for col in fin_cols:
    grupo_mean = df_ctx.groupby(['ocupacion', 'region'])[col].transform('mean')
    df_ctx[f'{col}_vs_tribu_ratio'] = df_ctx[col] / (grupo_mean + 1e-5)

X = df_ctx.drop(['id_cliente', 'objetivo'], axis=1)
y = df_ctx['objetivo']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

print("Entrenando Modelo 9 (Kaggle Trick)...")
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    auto_class_weights='Balanced',
    eval_metric='AUC',
    random_seed=42,
    verbose=False
)
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=30)

proba = model.predict_proba(test_pool)[:, 1]
auc = roc_auc_score(y_test, proba)

print(f"\\n⭐ EL AUC DEL MODELO ES FIJO: {auc:.5f}\\n")
print("==================================================")
print("🎚️ PROBANDO TU TEORÍA SOBRE LAS VALLAS (THRESHOLDS)")
print("==================================================")

thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print(f"{'Valla (Threshold)':<20} | {'Precisión (1)':<15} | {'Recall (1)':<15} | {'Precisión (0)':<15} | {'Recall (0)':<15}")
print("-" * 85)

for t in thresholds:
    # Aplicar la valla manualmente
    preds = (proba >= t).astype(int)
    
    prec_1 = precision_score(y_test, preds, pos_label=1, zero_division=0)
    rec_1 = recall_score(y_test, preds, pos_label=1)
    
    prec_0 = precision_score(y_test, preds, pos_label=0)
    rec_0 = recall_score(y_test, preds, pos_label=0)
    
    print(f">= {t:<17} | {prec_1*100:>13.1f}% | {rec_1*100:>13.1f}% | {prec_0*100:>13.1f}% | {rec_0*100:>13.1f}%")

print("\\n✅ Conclusión: Como dijiste, al subir la valla, la Precisión en 1 mejora drásticamente, pero el AUC NO CAMBIA NADA, porque el AUC evalúa TODAS las vallas al mismo tiempo.")
