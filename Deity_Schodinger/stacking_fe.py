"""
Stacking Ensemble Completo con Feature Engineering Temporal
=============================================================

Innovación: Stacking con meta-modelo usando OOF predictions

Arquitectura:
1. Feature Engineering Temporal (Lags, Rolling, Aggregations, Target Encoding)
2. Entrenar 3 modelos base (XGBoost, LightGBM, CatBoost) con FE
3. Generar OOF predictions con Time-Series CV
4. Crear meta-features: [pred_xgb, pred_lgb, pred_cat, pred_avg, pred_std, pred_min, pred_max]
5. Entrenar meta-modelo (LogisticRegression) con OOF predictions
6. Aplicar stacking a test
7. Generar submission final

Uso:
    python stacking_fe.py
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from oot_validation import OOTValidator
from temporal_feature_engineering import TemporalFeatureEngineer
from time_series_cv import OutOfTimeSplit


class StackingEnsembleFE:
    """
    Stacking Ensemble con Feature Engineering Temporal y OOF predictions.
    """
    
    def __init__(self):
        self.models = {}
        self.oof_predictions = {}
        self.meta_model = None
        self.time_cv = OutOfTimeSplit()
        
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
            feature_cols: List[str]) -> 'StackingEnsembleFE':
        """
        Entrenar el ensemble completo con FE y OOF predictions.
        """
        print("=" * 70)
        print("ENTRENANDO STACKING ENSEMBLE CON FE + OOF")
        print("=" * 70)
        
        # Combinar train y val para OOF
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        train_val_y = pd.concat([train_df['objetivo'], val_df['objetivo']], ignore_index=True)
        
        X = train_val_df[feature_cols].fillna(0)
        y = train_val_y
        
        # 1. Entrenar XGBoost con OOF
        print("\nEntrenando XGBoost con OOF predictions...")
        self.models['xgb'] = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        self.oof_predictions['xgb'] = self._get_oof_predictions(
            self.models['xgb'], X, y, feature_cols, train_val_df
        )
        print(f"XGBoost OOF AUC: {roc_auc_score(y, self.oof_predictions['xgb']):.4f}")
        
        # 2. Entrenar LightGBM con OOF
        print("\nEntrenando LightGBM con OOF predictions...")
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
        self.oof_predictions['lgb'] = self._get_oof_predictions(
            self.models['lgb'], X, y, feature_cols, train_val_df
        )
        print(f"LightGBM OOF AUC: {roc_auc_score(y, self.oof_predictions['lgb']):.4f}")
        
        # 3. Entrenar CatBoost con OOF
        print("\nEntrenando CatBoost con OOF predictions...")
        self.models['cat'] = cb.CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            depth=6,
            learning_rate=0.05,
            n_estimators=200,
            random_state=42,
            verbose=False
        )
        self.oof_predictions['cat'] = self._get_oof_predictions(
            self.models['cat'], X, y, feature_cols, train_val_df
        )
        print(f"CatBoost OOF AUC: {roc_auc_score(y, self.oof_predictions['cat']):.4f}")
        
        # 4. Crear meta-features
        print("\nCreando meta-features...")
        meta_train = self._create_meta_features(self.oof_predictions)
        
        # 5. Entrenar meta-modelo
        print("Entrenando meta-modelo (LogisticRegression)...")
        self.meta_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        self.meta_model.fit(meta_train, y)
        
        # Evaluar meta-modelo
        meta_oof = self.meta_model.predict_proba(meta_train)[:, 1]
        print(f"Meta-Modelo OOF AUC: {roc_auc_score(y, meta_oof):.4f}")
        
        # Evaluar promedio simple
        avg_oof = np.mean(list(self.oof_predictions.values()), axis=0)
        print(f"Promedio Simple OOF AUC: {roc_auc_score(y, avg_oof):.4f}")
        
        return self
    
    def _get_oof_predictions(self, model, X: pd.DataFrame, y: pd.Series, 
                            feature_cols: List[str], df: pd.DataFrame) -> np.ndarray:
        """
        Generar OOF predictions usando Time-Series CV.
        """
        oof_pred = np.zeros(len(X))
        
        # Usar Time-Series CV (máximo 5 folds para velocidad)
        folds = list(self.time_cv.split(df))[:5]
        
        for train_idx, val_idx in folds:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            
            model.fit(X_train, y_train)
            oof_pred[val_idx] = model.predict_proba(X_val)[:, 1]
        
        return oof_pred
    
    def _create_meta_features(self, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Crear meta-features a partir de OOF predictions."""
        meta = pd.DataFrame()
        
        for name, pred in predictions.items():
            meta[f'pred_{name}'] = pred
        
        # Agregar estadísticas
        preds = list(predictions.values())
        meta['pred_avg'] = np.mean(preds, axis=0)
        meta['pred_std'] = np.std(preds, axis=0)
        meta['pred_min'] = np.min(preds, axis=0)
        meta['pred_max'] = np.max(preds, axis=0)
        meta['pred_median'] = np.median(preds, axis=0)
        
        return meta
    
    def predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                feature_cols: List[str]) -> np.ndarray:
        """Generar predicciones finales re-entrenando con train completo."""
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['objetivo']
        X_test = test_df[feature_cols].fillna(0)
        
        # Re-entrenar modelos con train completo
        print("\nRe-entrenando modelos con train completo...")
        
        # Re-entrenar XGBoost
        print("Re-entrenando XGBoost completo...")
        self.models['xgb'].fit(X_train, y_train)
        test_predictions = {}
        test_predictions['xgb'] = self.models['xgb'].predict_proba(X_test)[:, 1]
        
        # Re-entrenar LightGBM
        print("Re-entrenando LightGBM completo...")
        self.models['lgb'].fit(X_train, y_train)
        test_predictions['lgb'] = self.models['lgb'].predict_proba(X_test)[:, 1]
        
        # Re-entrenar CatBoost
        print("Re-entrenando CatBoost completo...")
        self.models['cat'].fit(X_train, y_train)
        test_predictions['cat'] = self.models['cat'].predict_proba(X_test)[:, 1]
        
        # Crear meta-features
        meta_test = self._create_meta_features(test_predictions)
        
        # Predicción final
        return self.meta_model.predict_proba(meta_test)[:, 1]


def example_stacking_fe():
    """
    Stacking Ensemble completo con Feature Engineering Temporal.
    """
    print("=" * 70)
    print("STACKING ENSEMBLE - FE + OOF + META-MODELO")
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
    
    # Entrenar ensemble
    ensemble = StackingEnsembleFE()
    ensemble.fit(train_fe, val_fe, feature_cols)
    
    # Evaluar en validation
    print("\n" + "=" * 70)
    print("EVALUACIÓN EN VALIDATION")
    print("=" * 70)
    
    X_val = val_fe[feature_cols].fillna(0)
    y_val = val_fe['objetivo']
    
    for name, model in ensemble.models.items():
        model.fit(train_fe[feature_cols].fillna(0), train_fe['objetivo'])
        val_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        print(f"{name.upper()} Val AUC: {auc:.4f}")
    
    # Meta-modelo en validation
    meta_val = ensemble._create_meta_features({
        'xgb': ensemble.models['xgb'].predict_proba(X_val)[:, 1],
        'lgb': ensemble.models['lgb'].predict_proba(X_val)[:, 1],
        'cat': ensemble.models['cat'].predict_proba(X_val)[:, 1]
    })
    meta_pred = ensemble.meta_model.predict_proba(meta_val)[:, 1]
    meta_auc = roc_auc_score(y_val, meta_pred)
    print(f"META Val AUC: {meta_auc:.4f}")
    
    # Generar predicciones para test
    print("\n" + "=" * 70)
    print("GENERANDO PREDICCIONES PARA TEST")
    print("=" * 70)
    
    # Combinar train y val para re-entrenamiento final
    train_val_fe = pd.concat([train_fe, val_fe], ignore_index=True)
    train_val_y = pd.concat([train_fe['objetivo'], val_fe['objetivo']], ignore_index=True)
    
    test_pred = ensemble.predict(train_val_fe, test_fe, feature_cols)
    
    print(f"Predicciones generadas: {len(test_pred)}")
    print(f"Rango: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    print(f"Promedio: {test_pred.mean():.4f}")
    print(f"Std: {test_pred.std():.4f}")
    
    # Crear submission
    submission = pd.DataFrame({
        'id_cliente': test_df['id_cliente'],
        'prediccion': test_pred
    })
    
    submission_path = 'submission_stacking_fe.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission guardada en: {submission_path}")
    
    # Comparación
    print("\n" + "=" * 70)
    print("COMPARACIÓN CON CATBOOST INDIVIDUAL")
    print("=" * 70)
    print(f"CatBoost individual FE:  0.6229 / Gini 0.2458")
    print(f"Stacking FE + OOF:       {meta_auc:.4f} / Gini {2*meta_auc-1:.4f}")
    print(f"Mejora AUC:               {meta_auc - 0.6229:.4f}")
    print(f"Mejora Gini:              {(2*meta_auc-1) - 0.2458:.4f}")
    
    if meta_auc > 0.6229:
        print("\n[OK] MEJORA ALCANZADA")
    else:
        print(f"\n[X] Sin mejora: {meta_auc:.4f}")
    
    print("\n" + "=" * 70)
    print("FINALIZADO")
    print("=" * 70)
    
    return ensemble, submission


if __name__ == "__main__":
    ensemble, submission = example_stacking_fe()
