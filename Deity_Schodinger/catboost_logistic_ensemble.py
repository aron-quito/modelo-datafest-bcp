"""
CatBoost + Logistic Regression Ensemble
=======================================

Estrategia del diagnóstico #3:
- Usar modelo lineal (Logistic Regression) para diversificar predicciones
- Los árboles tienen sesgo de corte rígido
- La regresión logística tiene función de pérdida suave/lineal
- El ensemble combina ambos para mejor generalización

Uso:
    python catboost_logistic_ensemble.py
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import catboost as cb

from oot_validation import OOTValidator
from temporal_feature_engineering import TemporalFeatureEngineer


def example_catboost_logistic_ensemble():
    """
    Ensemble de CatBoost + Logistic Regression.
    """
    print("=" * 70)
    print("ENSEMBLE CATBOOST + LOGISTIC REGRESSION")
    print("=" * 70)
    
    # Cargar datos
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"\nTrain: {train_df.shape[0]} registros")
    print(f"Test: {test_df.shape[0]} registros")
    
    # Feature Engineering Temporal
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING TEMPORAL")
    print("=" * 70)
    
    fe = TemporalFeatureEngineer(
        time_col='mes',
        group_col='id_cliente',
        target_col='objetivo',
        lag_periods=[1, 2],
        rolling_windows=[3],
        alpha=10.0
    )
    
    print("Fit en train...")
    fe.fit(train_df, train_df['objetivo'])
    
    print("Transform train...")
    train_fe = fe.transform(train_df)
    
    print("Transform test...")
    test_fe = fe.transform(test_df)
    
    print(f"Train después de FE: {train_fe.shape[1]} columnas")
    print(f"Test después de FE: {test_fe.shape[1]} columnas")
    
    # Dividir OOT
    print("\n" + "=" * 70)
    print("DIVISIÓN OOT")
    print("=" * 70)
    
    oot_validator = OOTValidator()
    train, val, test = oot_validator.print_summary(train_df, test_df)
    
    # Aplicar FE a cada split
    print("\nAplicando FE a splits...")
    train_fe = fe.transform(train)
    val_fe = fe.transform(val)
    test_fe = fe.transform(test)
    
    # Seleccionar features numéricas
    feature_cols = [col for col in train_fe.columns 
                   if col not in ['id_cliente', 'mes', 'objetivo'] 
                   and train_fe[col].dtype in [np.float64, np.int64, bool]]
    
    # Asegurar consistencia
    missing_cols = set(feature_cols) - set(val_fe.columns)
    for col in missing_cols:
        val_fe[col] = 0
    
    missing_cols = set(feature_cols) - set(test_fe.columns)
    for col in missing_cols:
        test_fe[col] = 0
    
    print(f"\nFeatures seleccionadas: {len(feature_cols)}")
    
    # Entrenar CatBoost
    print("\n" + "=" * 70)
    print("ENTRENANDO CATBOOST")
    print("=" * 70)
    
    cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'depth': 6,
        'learning_rate': 0.05,
        'random_strength': 1.0,
        'bagging_temperature': 1.0,
        'l2_leaf_reg': 3.0,
        'border_count': 128,
        'n_estimators': 300,
        'random_state': 42,
        'verbose': False
    }
    
    cat_model = cb.CatBoostClassifier(**cat_params)
    
    X_train = train_fe[feature_cols].fillna(0)
    y_train = train_fe['objetivo']
    X_val = val_fe[feature_cols].fillna(0)
    y_val = val_fe['objetivo']
    
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    cat_pred = cat_model.predict_proba(X_val)[:, 1]
    cat_auc = roc_auc_score(y_val, cat_pred)
    print(f"CatBoost Val AUC: {cat_auc:.4f}")
    
    # Entrenar Logistic Regression (con regularización fuerte)
    print("\n" + "=" * 70)
    print("ENTRENANDO LOGISTIC REGRESSION")
    print("=" * 70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    log_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=0.1,  # Regularización fuerte
        penalty='l2',
        solver='lbfgs'
    )
    
    log_model.fit(X_train_scaled, y_train)
    
    log_pred = log_model.predict_proba(X_val_scaled)[:, 1]
    log_auc = roc_auc_score(y_val, log_pred)
    print(f"Logistic Regression Val AUC: {log_auc:.4f}")
    
    # Calcular correlación de predicciones
    correlation = np.corrcoef(cat_pred, log_pred)[0, 1]
    print(f"Correlación CatBoost vs Logistic: {correlation:.4f}")
    
    # Ensemble simple (promedio)
    print("\n" + "=" * 70)
    print("ENSEMBLE (PROMEDIO PONDERADO)")
    print("=" * 70)
    
    # Optimizar pesos del ensemble
    best_auc = 0
    best_weight = 0.5
    
    for w in np.arange(0, 1.1, 0.1):
        ensemble_pred = w * cat_pred + (1 - w) * log_pred
        auc = roc_auc_score(y_val, ensemble_pred)
        if auc > best_auc:
            best_auc = auc
            best_weight = w
    
    print(f"Mejor peso CatBoost: {best_weight:.1f}")
    print(f"Mejor peso Logistic: {1 - best_weight:.1f}")
    print(f"Ensemble Val AUC: {best_auc:.4f}")
    
    # Generar predicciones para test
    print("\n" + "=" * 70)
    print("GENERANDO PREDICCIONES PARA TEST")
    print("=" * 70)
    
    X_test = test_fe[feature_cols].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    
    # Re-entrenar con train completo
    train_val_fe = pd.concat([train_fe, val_fe], ignore_index=True)
    train_val_y = pd.concat([train_fe['objetivo'], val_fe['objetivo']], ignore_index=True)
    
    X_train_val = train_val_fe[feature_cols].fillna(0)
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    
    cat_model.fit(X_train_val, train_val_y, verbose=False)
    log_model.fit(X_train_val_scaled, train_val_y)
    
    cat_test_pred = cat_model.predict_proba(X_test)[:, 1]
    log_test_pred = log_model.predict_proba(X_test_scaled)[:, 1]
    
    ensemble_test_pred = best_weight * cat_test_pred + (1 - best_weight) * log_test_pred
    
    print(f"Predicciones generadas: {len(ensemble_test_pred)}")
    print(f"Rango: [{ensemble_test_pred.min():.4f}, {ensemble_test_pred.max():.4f}]")
    print(f"Promedio: {ensemble_test_pred.mean():.4f}")
    print(f"Std: {ensemble_test_pred.std():.4f}")
    
    # Crear submission
    submission = pd.DataFrame({
        'id_cliente': test_df['id_cliente'],
        'prediccion': ensemble_test_pred
    })
    
    submission_path = 'submission_catboost_logistic_ensemble.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission guardada en: {submission_path}")
    
    # Comparación
    print("\n" + "=" * 70)
    print("COMPARACIÓN")
    print("=" * 70)
    print(f"CatBoost individual:     0.6229 / Gini 0.2458")
    print(f"Logistic Regression:    {log_auc:.4f} / Gini {2*log_auc-1:.4f}")
    print(f"Ensemble:               {best_auc:.4f} / Gini {2*best_auc-1:.4f}")
    print(f"Mejora vs CatBoost:     {best_auc - 0.6229:.4f}")
    
    if best_auc > 0.6229:
        print("\n[OK] MEJORA ALCANZADA")
    else:
        print(f"\n[X] Sin mejora: {best_auc:.4f}")
    
    print("\n" + "=" * 70)
    print("FINALIZADO")
    print("=" * 70)
    
    return cat_model, log_model, submission


if __name__ == "__main__":
    cat_model, log_model, submission = example_catboost_logistic_ensemble()
