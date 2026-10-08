import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.makedirs('locuras/16_ternary_labeling', exist_ok=True)

print("1. CARGANDO DATOS")
df = pd.read_csv('dataset/train.csv')

# Preparar variables
df = df.drop(columns=['id_cliente', 'mes'])
cat_features = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
bool_features = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']

for col in bool_features:
    df[col] = df[col].astype(int)

X = df.drop(columns=['objetivo'])
y = df['objetivo']

# Guardamos el Validation puro con etiquetas originales (0 y 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

print("2. DETECTANDO LA ZONA DE SOLAPAMIENTO (OOF Predictions)")
# Entrenamos un CatBoost rápido con Cross Validation para que cada muestra 
# tenga una probabilidad calculada por un modelo que nunca la vio en su entrenamiento
oof_probs = np.zeros(len(X_train))
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_f_train, y_f_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_f_val, y_f_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # Modelo detector de solapamiento
    detector = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        cat_features=cat_features, auto_class_weights='Balanced',
        verbose=0, random_seed=42
    )
    detector.fit(X_f_train, y_f_train)
    oof_probs[val_idx] = detector.predict_proba(X_f_val)[:, 1]

print("3. CREANDO EL TERNARY LABELING (Clase 2 = Ambigüedad)")
# Definimos el margen de incertidumbre. 
# Si el modelo predijo probabilidad entre 0.35 y 0.65, es una zona muy gris.
umbral_inferior = 0.35
umbral_superior = 0.65

y_train_ternary = y_train.copy()
zona_gris = (oof_probs >= umbral_inferior) & (oof_probs <= umbral_superior)
y_train_ternary[zona_gris] = 2

distribucion = y_train_ternary.value_counts().to_dict()
print(f"Nueva distribución de clases en Entrenamiento:")
print(f"Clase 0 (Rechazo Seguro): {distribucion.get(0, 0)}")
print(f"Clase 1 (Compra Segura) : {distribucion.get(1, 0)}")
print(f"Clase 2 (Solapamiento)  : {distribucion.get(2, 0)}")

print("\n4. ENTRENANDO EL MODELO TERNARIO (MultiClass)")
# Ahora entrenamos el modelo final pero para predecir 3 CLASES
model_ternary = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    loss_function='MultiClass', # CRUCIAL: Cambio a MultiClase
    auto_class_weights='Balanced', # Balanceo automático para las 3 clases
    eval_metric='MultiClass',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

# OJO: Validamos con MultiClass, por lo que tenemos que crear temporalmente 
# un y_val_ternary ficticio (solo para el early stopping de la perdida multiclase)
# Asumiremos que el val es binario original, por lo que las loss se calcularan sobre 0 y 1
model_ternary.fit(
    X_train, y_train_ternary,
    eval_set=(X_val, y_val), # y_val no tiene clase 2, CatBoost lo acepta
    use_best_model=True
)

print("\n5. EVALUACIÓN Y DESENTRAÑAMIENTO")
# El modelo nos devuelve probabilidades para [Clase 0, Clase 1, Clase 2]
probs_val_ternary = model_ternary.predict_proba(X_val)

# Probabilidad pura de ser "1" (Compra Segura)
prob_clase_1 = probs_val_ternary[:, 1]
# Probabilidad de estar en la zona gris
prob_clase_2 = probs_val_ternary[:, 2]

# Para evaluar el AUC contra las etiquetas reales (0 y 1), 
# usamos la Probabilidad de Clase 1 estricta
auc_puro = roc_auc_score(y_val, prob_clase_1)

# ¿Qué pasa si sumamos la incertidumbre a favor de la clase 1? (Un enfoque más "valiente")
# Es decir, prob_1 + un porcentaje de prob_2
auc_hibrido = roc_auc_score(y_val, prob_clase_1 + (0.5 * prob_clase_2))

print(f"AUC (Usando solo Probabilidad de '1' Puro): {auc_puro:.4f}")
print(f"AUC (Usando Prob '1' + 50% de Prob 'Ambigua'): {auc_hibrido:.4f}")

# Sacamos un classification report usando la clase con mayor probabilidad
y_pred_ternary = np.argmax(probs_val_ternary, axis=1)

# Pero queremos ver como se comporta contra la realidad binaria
# Mapeamos las predicciones: si predice 2 (ambiguo), lo forzamos a 0 o 1 según 
# quien tenía la segunda probabilidad más alta, o lo analizamos por separado
y_pred_binario_final = np.where(
    y_pred_ternary == 2, 
    np.where(probs_val_ternary[:, 1] > probs_val_ternary[:, 0], 1, 0), 
    y_pred_ternary
)

print("Classification Report Final (Mapeando la Ambigüedad):")
print(classification_report(y_val, y_pred_binario_final))

# Guardar Resultados
results = {
    'distribucion_clases': distribucion,
    'auc_clase_1_pura': float(auc_puro),
    'auc_hibrido': float(auc_hibrido),
    'classification_report_mapeado': classification_report(y_val, y_pred_binario_final, output_dict=True)
}

with open('locuras/16_ternary_labeling/metricas.json', 'w') as f:
    json.dump(results, f, indent=4)

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, prob_clase_1)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Ternary (Puro 1) (AUC = {auc_puro:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Ternary Labeling')
plt.legend()
plt.savefig('locuras/16_ternary_labeling/roc_curve.png')
plt.close()

# Histograma de predicciones (Para ver la zona de ambigüedad)
plt.figure(figsize=(10,6))
sns.histplot(prob_clase_2, bins=50, kde=True, color='purple')
plt.title('Distribución de Probabilidad de ser "Zona Ambigua" (Clase 2) en Validación')
plt.xlabel('Probabilidad de Clase 2')
plt.savefig('locuras/16_ternary_labeling/hist_ambiguedad.png')
plt.close()

print("Proceso finalizado. Artefactos en locuras/16_ternary_labeling/")
