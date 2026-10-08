import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool
import shap
import cleanlab
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    os.makedirs('locuras/18_shap_and_cleanlab', exist_ok=True)

    print("1. CARGANDO DATOS")
    df = pd.read_csv('dataset/train.csv')

    df = df.drop(columns=['id_cliente', 'mes'])
    cat_features = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
    bool_features = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']

    for col in bool_features:
        df[col] = df[col].astype(int)

    X = df.drop(columns=['objetivo'])
    y = df['objetivo']

    print("2. CLEANLAB: BUSCANDO MENTIRAS EN EL DATASET (Label Errors)")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pred_probs = np.zeros((len(X), 2))

    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(iterations=300, depth=6, cat_features=cat_features, verbose=0, random_seed=42)
        model.fit(X_train, y_train)
        pred_probs[val_idx] = model.predict_proba(X_val)

    label_issues = cleanlab.filter.find_label_issues(
        labels=y.values,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence',
        n_jobs=1
    )

    print(f"¡Cleanlab ha encontrado {len(label_issues)} clientes que probablemente están ETIQUETADOS MAL!")

    print("\n3. SHAP: DESNUDANDO LAS COLUMNAS CON MAGIA NEGRA")
    model_final = CatBoostClassifier(iterations=300, depth=6, cat_features=cat_features, verbose=0, random_seed=42)
    model_final.fit(X, y)

    explainer = shap.TreeExplainer(model_final)
    shap_values = explainer.shap_values(X)

    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({'feature': X.columns, 'shap_importance': shap_sum})
    importance_df = importance_df.sort_values(by='shap_importance', ascending=False)

    print("\n--- LAS MEJORES COLUMNAS (MÁS SHAP) ---")
    print(importance_df.head(5))

    print("\n--- LA BASURA ABSOLUTA (MENOS SHAP) ---")
    print(importance_df.tail(10))

    plt.figure(figsize=(10,8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig('locuras/18_shap_and_cleanlab/shap_bar.png')
    plt.close()

    importance_df.to_csv('locuras/18_shap_and_cleanlab/shap_importances.csv', index=False)
    print("\n¡Análisis SHAP y Cleanlab finalizado! Revisa las imágenes.")

if __name__ == '__main__':
    main()
