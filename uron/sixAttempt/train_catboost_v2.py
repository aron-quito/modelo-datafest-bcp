import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import warnings
from feature_engineer import aplicar_ingenieria_variables

warnings.filterwarnings('ignore')

print("==================================================")
print("🚀 SEXTO INTENTO: CATBOOST + FEATURE ENGINEERING")
print("==================================================")

# 1. Cargar y procesar datos
df = pd.read_csv('../../dataset/train.csv')

# Aislar el 10% puro para la evaluación de la matriz de confusión real
# (Pero CatBoost usará el 90% en K-Fold para calcular el AUC oficial)
from sklearn.model_selection import train_test_split
X_all = df.drop(['id_cliente', 'objetivo'], axis=1)
y_all = df['objetivo']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.10, stratify=y_all, random_state=42
)

# Aplicar Ingeniería
print("\\nAplicando Feature Engineering a todo el dataset...")
X_train = aplicar_ingenieria_variables(X_train_raw)
X_test = aplicar_ingenieria_variables(X_test_raw)

# Extraer columnas categóricas después de la ingeniería
cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal', 'generacion', 'perfil_socioeconomico', 'perfil_riesgo_canal']

# Convertir tipos
for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# 2. Configurar K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

print("\\n🌲 Iniciando Entrenamiento CatBoost 5-Fold (Optimizando AUC)...")
models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\\n--- Fold {fold+1} ---")
    
    # Preparar Pools de CatBoost
    train_pool = Pool(X_train.iloc[train_idx], y_train.iloc[train_idx], cat_features=cat_cols)
    val_pool = Pool(X_train.iloc[val_idx], y_train.iloc[val_idx], cat_features=cat_cols)
    
    # Modelo CatBoost
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        early_stopping_rounds=50,
        thread_count=-1,
        verbose=100
    )
    
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    # Predecir OOF y Test
    oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
    test_preds += model.predict_proba(Pool(X_test, cat_features=cat_cols))[:, 1] / skf.n_splits
    models.append(model)

# 3. Resultados OOF (La métrica oficial de qué tan bien generaliza)
oof_auc = roc_auc_score(y_train, oof_preds)
print("\\n=======================================================")
print(f"⭐ OOF AUC CROSS-VALIDATION FINAL: {oof_auc:.5f}")
print("=======================================================")

if oof_auc > 0.631:
    print("🎉 ¡HEMOS ROTO EL TECHO DE CRISTAL! El Feature Engineering funcionó.")
else:
    print("⚠️ El AUC se mantuvo. La señal matemática es estrictamente limitada.")

# 4. Evaluación en Test Puro (Matriz de confusión)
threshold = np.percentile(test_preds, 85) # Top 15% como clase 1
test_preds_bin = (test_preds >= threshold).astype(int)

test_auc = roc_auc_score(y_test, test_preds)
print(f"\\n📊 AUC en Test Puro (10%): {test_auc:.5f}")
print("\\n📉 Matriz de Confusión (Test Puro):")
print(confusion_matrix(y_test, test_preds_bin))
print("\\n📋 Reporte de Clasificación:")
print(classification_report(y_test, test_preds_bin))

# 5. Importancia de las nuevas variables
feature_importances = models[0].get_feature_importance()
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\\n🔝 TOP 15 Variables Más Importantes (Descubiertas por el Árbol):")
print(importance_df.head(15).to_string(index=False))
