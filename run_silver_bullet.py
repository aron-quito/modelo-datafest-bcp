import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import cleanlab
import os
import json
import warnings
warnings.filterwarnings('ignore')

def main():
    os.makedirs('locuras/19_silver_bullet', exist_ok=True)

    print("1. CARGANDO DATOS Y CORTANDO LA GRASA (SHAP Drop)")
    df = pd.read_csv('dataset/train.csv')

    # IDs inútiles
    cols_to_drop = ['id_cliente', 'mes', 'objetivo']
    
    # La Basura (SHAP < 0.035)
    basura_shap = [
        'es_nuevo_cliente', 'tiene_seguro', 'dispositivo_principal', 
        'ocupacion', 'tiene_prestamo', 'region', 'dia_preferido_pago', 
        'visitas_web_ultimos_90_dias', 'canal_adquisicion', 'antiguedad_direccion_meses'
    ]
    
    X = df.drop(columns=cols_to_drop + basura_shap)
    y = df['objetivo']
    
    # Nuevas categóricas (las que sobrevivieron)
    cat_features = ['banda_riesgo']
    bool_features = ['tiene_tarjeta_credito', 'activo_movil']
    
    for col in bool_features:
        X[col] = X[col].astype(int)

    # 2. SPLIT STRICT (Protegiendo el Validation del Cleanlab)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    print("3. CLEANLAB: ENCONTRANDO MENTIRAS SOLO EN EL TRAIN")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pred_probs = np.zeros((len(X_train), 2))

    # Cross Validation para OOF (Out Of Fold) Probs
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_f_train, y_f_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_f_val, y_f_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Modelo rápido detector
        model_cv = CatBoostClassifier(
            iterations=300, depth=6, cat_features=cat_features, 
            verbose=0, random_seed=42, auto_class_weights='Balanced'
        )
        model_cv.fit(X_f_train, y_f_train)
        pred_probs[val_idx] = model_cv.predict_proba(X_f_val)

    # Cleanlab detecta los índices mentirosos
    label_issues = cleanlab.filter.find_label_issues(
        labels=y_train.values,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence',
        n_jobs=1
    )
    
    ceros_mentirosos = sum(y_train.iloc[label_issues] == 0)
    unos_mentirosos = sum(y_train.iloc[label_issues] == 1)

    print(f"¡Detectados {len(label_issues)} clientes corruptos!")
    print(f" -> {ceros_mentirosos} 'ceros' que actúan como unos.")
    print(f" -> {unos_mentirosos} 'unos' que actúan como ceros.")

    print("\n4. LA PURGA (Creando el Dataset de Platino)")
    # Eliminamos a los mentirosos del entrenamiento
    X_train_clean = X_train.drop(index=label_issues)
    y_train_clean = y_train.drop(index=label_issues)
    
    print(f"Distribución Final: {y_train_clean.value_counts().to_dict()}")

    print("\n5. ENTRENAMIENTO FINAL (The Silver Bullet)")
    # Entrenamos a CatBoost con el dataset hiper-destilado
    final_model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_features,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )

    final_model.fit(
        X_train_clean, y_train_clean,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    print("\n--- EVALUACIÓN EN EL MUNDO REAL (Validation Set) ---")
    y_pred = final_model.predict(X_val)
    y_prob = final_model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_prob)
    print(f"AUC: {auc:.4f}")
    print("Classification Report:")
    report = classification_report(y_val, y_pred)
    print(report)

    # Guardar Resultados
    results = {
        'auc': float(auc),
        'variables_basura_eliminadas': basura_shap,
        'label_errors_eliminados': len(label_issues),
        'ceros_mentirosos': int(ceros_mentirosos),
        'unos_mentirosos': int(unos_mentirosos),
        'distribucion_train_limpio': y_train_clean.value_counts().to_dict(),
        'classification_report': classification_report(y_val, y_pred, output_dict=True)
    }

    with open('locuras/19_silver_bullet/metricas.json', 'w') as f:
        json.dump(results, f, indent=4)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'Silver Bullet (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC - Cleanlab + SHAP Drop')
    plt.legend()
    plt.savefig('locuras/19_silver_bullet/roc_curve.png')
    plt.close()

    print("Proceso finalizado con éxito. Artefactos en locuras/19_silver_bullet/")

if __name__ == '__main__':
    main()
