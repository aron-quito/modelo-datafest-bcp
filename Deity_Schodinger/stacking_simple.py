"""
Stacking Ensemble Simple - Sin Feature Engineering Complejo
============================================================

Ensemble con Stacking usando:
- XGBoost, LightGBM, CatBoost
- Features numéricas básicas
- OOT Validation
- Meta-modelo LogisticRegression

Uso:
    python stacking_simple.py
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost no está instalado")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("Warning: LightGBM no está instalado")

try:
    import catboost as cb
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False
    print("Warning: CatBoost no está instalado")

from oot_validation import OOTValidator
from time_series_cv import OutOfTimeSplit


class SimpleStackingEnsemble:
    """
    Stacking Ensemble simple con OOT Validation.
    
    Flujo:
    1. Entrenar cada modelo base con OOT
    2. Generar predicciones en validation
    3. Crear meta-features
    4. Entrenar meta-modelo
    5. Generar predicciones finales
    """
    
    def __init__(self, use_xgb: bool = True, use_lgb: bool = True, use_cat: bool = True):
        self.use_xgb = use_xgb and XGB_AVAILABLE
        self.use_lgb = use_lgb and LGB_AVAILABLE
        self.use_cat = use_cat and CAT_AVAILABLE
        
        self.models = {}
        self.val_predictions = {}
        self.meta_model = None
        
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
            feature_cols: List[str]) -> 'SimpleStackingEnsemble':
        """
        Entrenar el ensemble con OOT.
        """
        print("=" * 70)
        print("ENTRENANDO STACKING ENSEMBLE")
        print("=" * 70)
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['objetivo']
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df['objetivo']
        
        # 1. Entrenar XGBoost
        if self.use_xgb:
            print("\nEntrenando XGBoost...")
            self.models['xgb'] = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                max_depth=6,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )
            self.models['xgb'].fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            self.val_predictions['xgb'] = self.models['xgb'].predict_proba(X_val)[:, 1]
            print(f"XGBoost Val AUC: {roc_auc_score(y_val, self.val_predictions['xgb']):.4f}")
        
        # 2. Entrenar LightGBM
        if self.use_lgb:
            print("\nEntrenando LightGBM...")
            self.models['lgb'] = lgb.LGBMClassifier(
                objective='binary',
                metric='auc',
                max_depth=6,
                learning_rate=0.05,
                n_estimators=200,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.models['lgb'].fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                                   callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            self.val_predictions['lgb'] = self.models['lgb'].predict_proba(X_val)[:, 1]
            print(f"LightGBM Val AUC: {roc_auc_score(y_val, self.val_predictions['lgb']):.4f}")
        
        # 3. Entrenar CatBoost
        if self.use_cat:
            print("\nEntrenando CatBoost...")
            self.models['cat'] = cb.CatBoostClassifier(
                loss_function='Logloss',
                eval_metric='AUC',
                depth=6,
                learning_rate=0.05,
                n_estimators=200,
                random_state=42,
                verbose=False
            )
            self.models['cat'].fit(X_train, y_train, eval_set=(X_val, y_val))
            self.val_predictions['cat'] = self.models['cat'].predict_proba(X_val)[:, 1]
            print(f"CatBoost Val AUC: {roc_auc_score(y_val, self.val_predictions['cat']):.4f}")
        
        # 4. Crear meta-features
        print("\nCreando meta-features...")
        meta_train = self._create_meta_features(self.val_predictions)
        
        # 5. Entrenar meta-modelo
        print("Entrenando meta-modelo (LogisticRegression)...")
        self.meta_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        self.meta_model.fit(meta_train, y_val)
        
        # Evaluar meta-modelo
        meta_pred = self.meta_model.predict_proba(meta_train)[:, 1]
        print(f"Meta-Modelo Val AUC: {roc_auc_score(y_val, meta_pred):.4f}")
        
        return self
    
    def _create_meta_features(self, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Crear meta-features a partir de predicciones."""
        meta = pd.DataFrame()
        
        for name, pred in predictions.items():
            meta[f'pred_{name}'] = pred
        
        # Agregar promedio
        if len(predictions) > 1:
            preds = list(predictions.values())
            meta['pred_avg'] = np.mean(preds, axis=0)
        
        return meta
    
    def predict(self, test_df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Generar predicciones finales."""
        X_test = test_df[feature_cols].fillna(0)
        
        # Generar predicciones de cada modelo
        test_predictions = {}
        
        if self.use_xgb and 'xgb' in self.models:
            test_predictions['xgb'] = self.models['xgb'].predict_proba(X_test)[:, 1]
        
        if self.use_lgb and 'lgb' in self.models:
            test_predictions['lgb'] = self.models['lgb'].predict_proba(X_test)[:, 1]
        
        if self.use_cat and 'cat' in self.models:
            test_predictions['cat'] = self.models['cat'].predict_proba(X_test)[:, 1]
        
        # Crear meta-features
        meta_test = self._create_meta_features(test_predictions)
        
        # Predicción final
        return self.meta_model.predict_proba(meta_test)[:, 1]


def example_stacking_simple():
    """
    Stacking Ensemble simple sin feature engineering complejo.
    """
    print("=" * 70)
    print("STACKING ENSEMBLE SIMPLE - BASELINE")
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
    
    # Entrenar ensemble
    ensemble = SimpleStackingEnsemble(
        use_xgb=True,
        use_lgb=LGB_AVAILABLE,
        use_cat=CAT_AVAILABLE
    )
    
    ensemble.fit(train, val, feature_cols)
    
    # Generar predicciones para test
    print("\n" + "=" * 70)
    print("GENERANDO PREDICCIONES PARA TEST")
    print("=" * 70)
    
    test_pred = ensemble.predict(test, feature_cols)
    
    print(f"Predicciones generadas: {len(test_pred)}")
    print(f"Rango: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    print(f"Promedio: {test_pred.mean():.4f}")
    
    # Crear submission
    submission = pd.DataFrame({
        'id_cliente': test_df['id_cliente'],
        'prediccion': test_pred
    })
    
    submission_path = 'submission_stacking_simple.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission guardada en: {submission_path}")
    
    # Comparación con modelos individuales
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 70)
    
    X_val = val[feature_cols].fillna(0)
    y_val = val['objetivo']
    
    for name, model in ensemble.models.items():
        pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, pred)
        print(f"{name.upper():>10} AUC: {auc:.4f}")
    
    # Meta-modelo
    meta_val = ensemble._create_meta_features(ensemble.val_predictions)
    meta_pred = ensemble.meta_model.predict_proba(meta_val)[:, 1]
    meta_auc = roc_auc_score(y_val, meta_pred)
    print(f"{'META':>10} AUC: {meta_auc:.4f}")
    
    # Promedio simple
    if len(ensemble.val_predictions) > 1:
        avg_pred = np.mean(list(ensemble.val_predictions.values()), axis=0)
        avg_auc = roc_auc_score(y_val, avg_pred)
        print(f"{'AVERAGE':>10} AUC: {avg_auc:.4f}")
    
    print("\n" + "=" * 70)
    print("FINALIZADO")
    print("=" * 70)
    
    return ensemble, submission


if __name__ == "__main__":
    ensemble, submission = example_stacking_simple()
