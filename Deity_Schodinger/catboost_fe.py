"""
CatBoost con Feature Engineering Temporal
=========================================

Mejor combinación para obtener AUC > 0.6:
- CatBoost (mejor modelo individual: 0.5699)
- Feature Engineering Temporal completo
- OOT Validation

Uso:
    python catboost_fe.py
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import catboost as cb

from oot_validation import OOTValidator
from temporal_feature_engineering import TemporalFeatureEngineer


def example_catboost_fe():
    """
    CatBoost con Feature Engineering Temporal completo.
    """
    print("=" * 70)
    print("CATBOOST + FEATURE ENGINEERING TEMPORAL")
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
        lag_periods=[1, 2],  # Reducido para velocidad
        rolling_windows=[3],  # Reducido para velocidad
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
    
    # Configurar CatBoost
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
    
    model = cb.CatBoostClassifier(**cat_params)
    
    # Entrenar
    X_train = train_fe[feature_cols].fillna(0)
    y_train = train_fe['objetivo']
    X_val = val_fe[feature_cols].fillna(0)
    y_val = val_fe['objetivo']
    
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Target balance (train): {y_train.mean():.4f}")
    print(f"Target balance (val): {y_val.mean():.4f}")
    
    print("\nEntrenando con early stopping...")
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Evaluar
    print("\n" + "-" * 70)
    print("MÉTRICAS DE VALIDACIÓN")
    print("-" * 70)
    
    val_pred = model.predict_proba(X_val)[:, 1]
    val_pred_class = (val_pred > 0.5).astype(int)
    
    metrics = {
        'AUC': roc_auc_score(y_val, val_pred),
        'Accuracy': accuracy_score(y_val, val_pred_class),
        'Precision': precision_score(y_val, val_pred_class, zero_division=0),
        'Recall': recall_score(y_val, val_pred_class, zero_division=0),
        'F1': f1_score(y_val, val_pred_class, zero_division=0)
    }
    
    for metric, value in metrics.items():
        print(f"{metric:>10}: {value:.4f}")
    
    # Feature importance
    print("\n" + "-" * 70)
    print("TOP 20 FEATURE IMPORTANCE")
    print("-" * 70)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance.head(20).to_string(index=False))
    
    # Generar predicciones para test
    print("\n" + "=" * 70)
    print("GENERANDO PREDICCIONES PARA TEST")
    print("=" * 70)
    
    X_test = test_fe[feature_cols].fillna(0)
    test_pred = model.predict_proba(X_test)[:, 1]
    
    print(f"Predicciones generadas: {len(test_pred)}")
    print(f"Rango: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    print(f"Promedio: {test_pred.mean():.4f}")
    print(f"Std: {test_pred.std():.4f}")
    
    # Crear submission
    submission = pd.DataFrame({
        'id_cliente': test_df['id_cliente'],
        'prediccion': test_pred
    })
    
    submission_path = 'submission_catboost_fe.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission guardada en: {submission_path}")
    
    # Comparación con baseline
    print("\n" + "=" * 70)
    print("COMPARACIÓN CON BASELINE")
    print("=" * 70)
    print(f"CatBoost sin FE:  0.5699")
    print(f"CatBoost con FE:   {metrics['AUC']:.4f}")
    print(f"Mejora:            {metrics['AUC'] - 0.5699:.4f}")
    
    if metrics['AUC'] > 0.6:
        print("\n[OK] OBJETIVO ALCANZADO: AUC > 0.6")
    else:
        print(f"\n[X] AUC aun por debajo de 0.6: {metrics['AUC']:.4f}")
    
    print("\n" + "=" * 70)
    print("FINALIZADO")
    print("=" * 70)
    
    return model, train_fe, val_fe, test_fe, submission


if __name__ == "__main__":
    model, train_fe, val_fe, test_fe, submission = example_catboost_fe()
