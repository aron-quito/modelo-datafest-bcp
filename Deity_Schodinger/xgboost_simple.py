"""
XGBoost Simple - Sin Feature Engineering Complejo
===================================================

Baseline rápido de XGBoost con:
- Features numéricas básicas
- OOT Validation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Error: XGBoost no está instalado. Instala con: pip install xgboost")
    exit(1)

from oot_validation import OOTValidator, OOTPipeline


def example_xgboost_simple():
    """
    XGBoost Simple sin feature engineering complejo.
    """
    print("=" * 70)
    print("XGBOOST SIMPLE - BASELINE")
    print("=" * 70)
    
    # Cargar datos
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"\nTrain: {train_df.shape[0]} registros")
    print(f"Test: {test_df.shape[0]} registros")
    
    # Dividir OOT
    print("\n" + "=" * 70)
    print("DIVISIÓN OOT")
    print("=" * 70)
    
    oot_validator = OOTValidator()
    train, val, test = oot_validator.print_summary(train_df, test_df)
    
    # Seleccionar features numéricas simples
    feature_cols = [col for col in train.columns 
                   if col not in ['id_cliente', 'mes', 'objetivo', 'ocupacion', 
                                 'region', 'canal_adquisicion', 'banda_riesgo', 
                                 'dispositivo_principal']
                   and train[col].dtype in [np.float64, np.int64, bool]]
    
    print(f"\nFeatures seleccionadas: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    # Configurar XGBoost
    print("\n" + "=" * 70)
    print("ENTRENANDO XGBOOST")
    print("=" * 70)
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 200,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    model = xgb.XGBClassifier(**xgb_params)
    
    # Entrenar
    X_train = train[feature_cols].fillna(0)
    y_train = train['objetivo']
    X_val = val[feature_cols].fillna(0)
    y_val = val['objetivo']
    
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Target balance (train): {y_train.mean():.4f}")
    print(f"Target balance (val): {y_val.mean():.4f}")
    
    print("\nEntrenando...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
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
    print("TOP 10 FEATURE IMPORTANCE")
    print("-" * 70)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance.head(10).to_string(index=False))
    
    # Generar predicciones para test
    print("\n" + "=" * 70)
    print("GENERANDO PREDICCIONES PARA TEST")
    print("=" * 70)
    
    X_test = test[feature_cols].fillna(0)
    test_pred = model.predict_proba(X_test)[:, 1]
    
    print(f"Predicciones generadas: {len(test_pred)}")
    print(f"Rango: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    print(f"Promedio: {test_pred.mean():.4f}")
    
    # Crear submission
    submission = pd.DataFrame({
        'id_cliente': test_df['id_cliente'],
        'prediccion': test_pred
    })
    
    submission_path = 'submission_xgboost_simple.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission guardada en: {submission_path}")
    
    print("\n" + "=" * 70)
    print("FINALIZADO")
    print("=" * 70)
    
    return model, submission


if __name__ == "__main__":
    model, submission = example_xgboost_simple()
