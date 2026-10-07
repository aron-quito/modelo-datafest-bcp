import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings('ignore')

print("==================================================")
print("🌐 EXPERIMENTO FINAL: CONTEXTO SOCIAL (TRIBU)")
print("==================================================")

df = pd.read_csv('../../dataset/train.csv')
y = df['objetivo']

# --- EXPERIMENTO 1: SÓLO CONTEXTO SOCIAL ---
print("\\n[Experimento 1] Calculando Contexto Social (Gana más/menos que su Tribu)...")
df_ctx = df.copy()

# Variables financieras a comparar con la tribu
fin_cols = ['ingresos', 'saldo_promedio', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses']

# Agrupar por Ocupación + Región para definir la "Tribu" del cliente
for col in fin_cols:
    grupo_mean = df_ctx.groupby(['ocupacion', 'region'])[col].transform('mean')
    df_ctx[f'{col}_vs_tribu_ratio'] = df_ctx[col] / (grupo_mean + 1e-5)
    df_ctx[f'{col}_vs_tribu_diff'] = df_ctx[col] - grupo_mean

# --- EXPERIMENTO 2: CONTEXTO SOCIAL + FEATURE ENGINEERING ORIGINAL ---
print("[Experimento 2] Añadiendo Feature Engineering original (Ratios Personales)...")
df_full = df_ctx.copy()

# Feature Engineering Clásico
df_full['deuda_estimada'] = df_full['ingresos'] * df_full['ratio_deuda_ingresos']
df_full['ratio_saldo_ingreso'] = df_full['saldo_promedio'] / (df_full['ingresos'] + 1)
df_full['ratio_saldo_deuda'] = df_full['saldo_promedio'] / (df_full['deuda_estimada'] + 1)
df_full['dias_desde_actividad'] = df_full[['dias_ultima_interaccion', 'dias_ultima_transaccion']].min(axis=1)
df_full['brecha_actividad'] = (df_full['dias_ultima_interaccion'] - df_full['dias_ultima_transaccion']).abs()
df_full['perfil_riesgo_canal'] = df_full['banda_riesgo'].astype(str) + '_' + df_full['canal_adquisicion'].astype(str)
df_full['region_dispositivo'] = df_full['region'].astype(str) + '_' + df_full['dispositivo_principal'].astype(str)

def evaluar_dataset(data, nombre_experimento):
    X = data.drop(['id_cliente', 'objetivo'], axis=1)
    
    # Manejar categóricas (incluyendo las nuevas como perfil_riesgo_canal)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype(str)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        eval_metric='AUC',
        auto_class_weights='Balanced', # Kaggle Trick activado
        random_seed=42,
        verbose=False
    )

    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

    preds = model.predict(test_pool)
    proba = model.predict_proba(test_pool)[:, 1]
    auc = roc_auc_score(y_test, proba)

    print(f"\\n--- RESULTADOS: {nombre_experimento} ---")
    print(classification_report(y_test, preds, digits=3))
    print(f"⭐ AUC FINAL: {auc:.5f}")

print("\\n🌲 Entrenando modelos (Kaggle Trick Activado)...")
evaluar_dataset(df_ctx, "1. SÓLO Contexto Social")
evaluar_dataset(df_full, "2. Contexto Social + Feature Engineering")
