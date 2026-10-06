"""
CatBoost Mejorado - Features para Romper el Sesgo de Riesgo
==============================================================

Basado en el diagnóstico del repositorio anterior:
- El modelo penaliza excesivamente a clientes de riesgo medium/high
- Necesitamos features específicas para clientes de baja vinculación/riesgo medio

Nuevas Features:
1. visitas_web_por_producto = visitas_web / (numero_productos + 1)
2. saldo_por_dias_inactividad = saldo / (dias_ultima_transaccion + 1)
3. enganche_por_riesgo = (activo_movil * ingresos) / (riesgo_score + 1)
4. producto_riesgo_ratio = numero_productos / (riesgo_score + 1)
5. delta_visitas = visitas_web - rolling_mean_visitas

Uso:
    python catboost_improved.py
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import catboost as cb

from oot_validation import OOTValidator
from temporal_feature_engineering import TemporalFeatureEngineer


class ImprovedFeatureEngineer(TemporalFeatureEngineer):
    """
    Feature Engineer mejorado con features específicas para romper sesgo de riesgo.
    """
    
    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agregar features domain-specific mejorados.
        """
        df = super()._add_domain_features(df)
        
        # Nuevas features específicas para clientes de riesgo medio/alto
        
        # 1. visitas_web_por_producto = visitas_web / (numero_productos + 1)
        if 'visitas_web_ultimos_90_dias' in df.columns and 'numero_productos' in df.columns:
            df['visitas_web_por_producto'] = df['visitas_web_ultimos_90_dias'] / (df['numero_productos'] + 1)
        
        # 2. saldo_por_dias_inactividad = saldo / (dias_ultima_transaccion + 1)
        if 'saldo_promedio' in df.columns and 'dias_ultima_transaccion' in df.columns:
            df['saldo_por_dias_inactividad'] = df['saldo_promedio'] / (df['dias_ultima_transaccion'] + 1)
        
        # 3. enganche_por_riesgo = (activo_movil * ingresos) / (riesgo_score + 1)
        if 'activo_movil' in df.columns and 'ingresos' in df.columns:
            # Crear score de riesgo numérico
            riesgo_map = {'low': 1, 'medium': 2, 'high': 3}
            if 'banda_riesgo' in df.columns:
                df['riesgo_score'] = df['banda_riesgo'].map(riesgo_map).fillna(2)
                df['enganche_por_riesgo'] = (df['activo_movil'].astype(int) * df['ingresos']) / (df['riesgo_score'] + 1)
        
        # 4. producto_riesgo_ratio = numero_productos / (riesgo_score + 1)
        if 'numero_productos' in df.columns and 'riesgo_score' in df.columns:
            df['producto_riesgo_ratio'] = df['numero_productos'] / (df['riesgo_score'] + 1)
        
        # 5. Ratio deuda/ingresos ponderado por riesgo
        if 'ratio_deuda_ingresos' in df.columns and 'riesgo_score' in df.columns:
            df['deuda_riesgo_ratio'] = df['ratio_deuda_ingresos'] / df['riesgo_score']
        
        # 6. Features de interacción para clientes de baja vinculación
        if 'saldo_promedio' in df.columns and 'numero_productos' in df.columns:
            df['saldo_por_producto'] = df['saldo_promedio'] / (df['numero_productos'] + 1)
        
        # 7. Actividad reciente normalizada por antigüedad
        if 'dias_ultima_transaccion' in df.columns and 'antiguedad_cuenta_meses' in df.columns:
            df['actividad_normalizada'] = df['dias_ultima_transaccion'] / (df['antiguedad_cuenta_meses'] + 1)
        
        return df


def example_catboost_improved():
    """
    CatBoost con features mejoradas para romper sesgo de riesgo.
    """
    print("=" * 70)
    print("CATBOOST MEJORADO - FEATURES ANTI-SESGO RIESGO")
    print("=" * 70)
    
    # Cargar datos
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"\nTrain: {train_df.shape[0]} registros")
    print(f"Test: {test_df.shape[0]} registros")
    
    # Feature Engineering Temporal Mejorado
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING TEMPORAL MEJORADO")
    print("=" * 70)
    
    fe = ImprovedFeatureEngineer(
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
    
    # Configurar CatBoost con regularización más fuerte en banda_riesgo
    print("\n" + "=" * 70)
    print("ENTRENANDO CATBOOST MEJORADO")
    print("=" * 70)
    
    cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'depth': 7,  # Un poco más profundo
        'learning_rate': 0.03,  # Más lento para más precisión
        'random_strength': 1.5,  # Más aleatoriedad para evitar overfitting
        'bagging_temperature': 1.5,
        'l2_leaf_reg': 5.0,  # Más regularización
        'border_count': 128,
        'n_estimators': 500,  # Más árboles
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
        early_stopping_rounds=100,
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
    
    submission_path = 'submission_catboost_improved.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission guardada en: {submission_path}")
    
    # Comparación
    print("\n" + "=" * 70)
    print("COMPARACIÓN CON CATBOOST ORIGINAL")
    print("=" * 70)
    print(f"CatBoost original FE:  0.6229 / Gini 0.2458")
    print(f"CatBoost improved FE:  {metrics['AUC']:.4f} / Gini {2*metrics['AUC']-1:.4f}")
    print(f"Mejora AUC:              {metrics['AUC'] - 0.6229:.4f}")
    print(f"Mejora Gini:             {(2*metrics['AUC']-1) - 0.2458:.4f}")
    
    if metrics['AUC'] > 0.6229:
        print("\n[OK] MEJORA ALCANZADA")
    else:
        print(f"\n[X] Sin mejora: {metrics['AUC']:.4f}")
    
    print("\n" + "=" * 70)
    print("FINALIZADO")
    print("=" * 70)
    
    return model, train_fe, val_fe, test_fe, submission


if __name__ == "__main__":
    model, train_fe, val_fe, test_fe, submission = example_catboost_improved()
