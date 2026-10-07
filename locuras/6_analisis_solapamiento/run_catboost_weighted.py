import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings('ignore')

print("==================================================")
print("🎯 KAGGLE TRICK: CATBOOST CON PESOS BALANCEADOS")
print("==================================================")

df = pd.read_csv('../../dataset/train.csv')
X = df.drop(['id_cliente', 'objetivo'], axis=1)
y = df['objetivo']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

print("🌲 Entrenando CatBoost obligándolo a respetar a la Clase 1...")
# auto_class_weights='Balanced' penaliza fuertemente equivocarse con la clase minoritaria
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    eval_metric='AUC',
    auto_class_weights='Balanced', # El secreto de Kaggle
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

preds = model.predict(test_pool)
proba = model.predict_proba(test_pool)[:, 1]

auc = roc_auc_score(y_test, proba)

print("\\n=======================================================")
print("🎯 REPORTE DE EXACTITUD POR CLASE (Con Pesos)")
print("=======================================================")
print(classification_report(y_test, preds, digits=3))
print(f"⭐ AUC FINAL (Kaggle Trick): {auc:.5f}")
print("=======================================================")
