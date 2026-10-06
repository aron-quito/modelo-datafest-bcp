"""
Out-of-Time (OOT) Validation - Estrategia de Competencia
===========================================================

Estrategia: Train (202601-202610) → Val (202611) → Test (202612)
Simula exactamente el escenario de competencia.

Más conservador que CV estándar, pero más realista.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


class OOTValidator:
    """
    Out-of-Time Validator para competiciones de machine learning.
    
    Divide los datos en:
    - Train: Meses históricos (202601-202610)
    - Validation: Mes más reciente de train (202611)
    - Test: Mes futuro de competencia (202612)
    
    Esta estrategia simula exactamente el escenario real de competencia
    donde entrenas con datos históricos y debes predecir el mes siguiente.
    """
    
    def __init__(self, train_months: List[int] = None, val_month: int = None, 
                 test_month: int = None, time_col: str = 'mes'):
        """
        Inicializar el OOT Validator.
        
        Parámetros:
        -----------
        train_months : List[int]
            Lista de meses para entrenamiento (default: 202601-202610)
        val_month : int
            Mes para validación (default: 202611)
        test_month : int
            Mes para test/competencia (default: 202612)
        time_col : str
            Nombre de la columna temporal (default: 'mes')
        """
        self.time_col = time_col
        
        # Configuración por defecto para este proyecto
        if train_months is None:
            self.train_months = list(range(202601, 202611))  # 202601-202610
        else:
            self.train_months = train_months
            
        if val_month is None:
            self.val_month = 202611
        else:
            self.val_month = val_month
            
        if test_month is None:
            self.test_month = 202612
        else:
            self.test_month = test_month
    
    def split_train_val(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en train y validation.
        
        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame con columna temporal
            
        Retorna:
        --------
        (train_df, val_df) : Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames de entrenamiento y validación
        """
        train_df = df[df[self.time_col].isin(self.train_months)].copy()
        val_df = df[df[self.time_col] == self.val_month].copy()
        
        print(f"Train: {len(train_df)} registros (meses {self.train_months[0]}-{self.train_months[-1]})")
        print(f"Validation: {len(val_df)} registros (mes {self.val_month})")
        
        return train_df, val_df
    
    def get_test_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtiene los datos de test (mes de competencia).
        
        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame con columna temporal
            
        Retorna:
        --------
        test_df : pd.DataFrame
            DataFrame de test
        """
        test_df = df[df[self.time_col] == self.test_month].copy()
        print(f"Test: {len(test_df)} registros (mes {self.test_month})")
        
        return test_df
    
    def split_all(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide train en train/val y retorna test por separado.
        
        Parámetros:
        -----------
        train_df : pd.DataFrame
            DataFrame de train (con meses históricos)
        test_df : pd.DataFrame
            DataFrame de test (mes de competencia)
            
        Retorna:
        --------
        (train, val, test) : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            DataFrames de train, validation y test
        """
        train, val = self.split_train_val(train_df)
        test = self.get_test_data(test_df)
        
        return train, val, test
    
    def print_summary(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Imprime un resumen de la división OOT.
        """
        print("=" * 70)
        print("OUT-OF-TIME VALIDATION - ESTRATEGIA DE COMPETENCIA")
        print("=" * 70)
        
        train, val = self.split_train_val(train_df)
        test = self.get_test_data(test_df)
        
        print("\n" + "-" * 70)
        print("RESUMEN DE DIVISIÓN")
        print("-" * 70)
        print(f"Train:      {len(train):>6} registros ({len(train)/(len(train)+len(val)+len(test))*100:.1f}%)")
        print(f"Validation: {len(val):>6} registros ({len(val)/(len(train)+len(val)+len(test))*100:.1f}%)")
        print(f"Test:       {len(test):>6} registros ({len(test)/(len(train)+len(val)+len(test))*100:.1f}%)")
        print("-" * 70)
        print(f"Total:      {len(train)+len(val)+len(test):>6} registros")
        print("=" * 70)
        
        return train, val, test


class OOTPipeline:
    """
    Pipeline completo para entrenamiento con OOT validation.
    
    Incluye:
    - División OOT de datos
    - Entrenamiento de modelo
    - Evaluación en validation
    - Generación de predicciones para test
    """
    
    def __init__(self, model: BaseEstimator, oot_validator: OOTValidator,
                 feature_cols: List[str], target_col: str = 'objetivo'):
        """
        Inicializar el pipeline OOT.
        
        Parámetros:
        -----------
        model : BaseEstimator
            Modelo de sklearn a entrenar
        oot_validator : OOTValidator
            Validador OOT configurado
        feature_cols : List[str]
            Lista de features a usar
        target_col : str
            Nombre de la columna target
        """
        self.model = model
        self.oot_validator = oot_validator
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.trained_model = None
        self.val_metrics = {}
        
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Entrena el modelo con datos de train y evalúa en validation.
        
        Parámetros:
        -----------
        train_df : pd.DataFrame
            DataFrame de entrenamiento
        val_df : pd.DataFrame
            DataFrame de validación
        """
        print("\n" + "=" * 70)
        print("ENTRENANDO MODELO CON OOT VALIDATION")
        print("=" * 70)
        
        # Preparar datos
        X_train = train_df[self.feature_cols].fillna(0)
        y_train = train_df[self.target_col]
        
        X_val = val_df[self.feature_cols].fillna(0)
        y_val = val_df[self.target_col]
        
        print(f"\nFeatures: {len(self.feature_cols)}")
        print(f"Train samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")
        print(f"Target balance (train): {y_train.mean():.4f}")
        print(f"Target balance (val): {y_val.mean():.4f}")
        
        # Entrenar modelo
        print("\nEntrenando modelo...")
        self.model.fit(X_train, y_train)
        self.trained_model = self.model
        
        # Evaluar en validation
        print("Evaluando en validation...")
        val_pred = self.model.predict_proba(X_val)[:, 1]
        val_pred_class = (val_pred > 0.5).astype(int)
        
        # Calcular métricas
        self.val_metrics = {
            'auc': roc_auc_score(y_val, val_pred),
            'accuracy': accuracy_score(y_val, val_pred_class),
            'precision': precision_score(y_val, val_pred_class, zero_division=0),
            'recall': recall_score(y_val, val_pred_class, zero_division=0),
            'f1': f1_score(y_val, val_pred_class, zero_division=0)
        }
        
        print("\n" + "-" * 70)
        print("MÉTRICAS DE VALIDACIÓN")
        print("-" * 70)
        for metric, value in self.val_metrics.items():
            print(f"{metric.upper():>10}: {value:.4f}")
        print("-" * 70)
        
        return self
    
    def predict_test(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones para el set de test.
        
        Parámetros:
        -----------
        test_df : pd.DataFrame
            DataFrame de test
            
        Retorna:
        --------
        predictions : np.ndarray
            Predicciones de probabilidad
        """
        if self.trained_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a fit() primero.")
        
        print("\n" + "=" * 70)
        print("GENERANDO PREDICCIONES PARA TEST")
        print("=" * 70)
        
        X_test = test_df[self.feature_cols].fillna(0)
        test_pred = self.model.predict_proba(X_test)[:, 1]
        
        print(f"Predicciones generadas: {len(test_pred)}")
        print(f"Rango de predicciones: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
        print(f"Promedio de predicciones: {test_pred.mean():.4f}")
        
        return test_pred
    
    def create_submission(self, test_df: pd.DataFrame, predictions: np.ndarray,
                         id_col: str = 'id_cliente', pred_col: str = 'prediccion') -> pd.DataFrame:
        """
        Crea el DataFrame de submission para la competencia.
        
        Parámetros:
        -----------
        test_df : pd.DataFrame
            DataFrame de test original
        predictions : np.ndarray
            Predicciones del modelo
        id_col : str
            Nombre de la columna ID
        pred_col : str
            Nombre de la columna de predicción
            
        Retorna:
        --------
        submission : pd.DataFrame
            DataFrame listo para submission
        """
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            pred_col: predictions
        })
        
        print("\n" + "=" * 70)
        print("SUBMISSION CREADA")
        print("=" * 70)
        print(f"Shape: {submission.shape}")
        print(f"Columnas: {list(submission.columns)}")
        print(f"\nPrimeras filas:")
        print(submission.head())
        
        return submission


def example_oot_validation():
    """
    Ejemplo completo de OOT validation para la competencia.
    """
    print("=" * 70)
    print("EJEMPLO COMPLETO: OOT VALIDATION")
    print("=" * 70)
    
    # Cargar datos
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"\nTrain cargado: {train_df.shape[0]} registros")
    print(f"Test cargado: {test_df.shape[0]} registros")
    print(f"Rango de meses train: {train_df['mes'].min()} - {train_df['mes'].max()}")
    print(f"Mes test: {test_df['mes'].unique()}")
    
    # Configurar OOT Validator
    oot_validator = OOTValidator(
        train_months=list(range(202601, 202611)),  # 202601-202610
        val_month=202611,
        test_month=202612,
        time_col='mes'
    )
    
    # Dividir datos
    train, val, test = oot_validator.print_summary(train_df, test_df)
    
    # Preparar features
    feature_cols = [col for col in train.columns if col not in ['id_cliente', 'mes', 'objetivo']]
    feature_cols = [col for col in feature_cols if train[col].dtype in [np.int64, np.float64, bool]]
    
    print(f"\nFeatures numéricas seleccionadas: {len(feature_cols)}")
    print(f"Features: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"Features: {feature_cols}")
    
    # Crear y entrenar pipeline
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    pipeline = OOTPipeline(
        model=model,
        oot_validator=oot_validator,
        feature_cols=feature_cols,
        target_col='objetivo'
    )
    
    pipeline.fit(train, val)
    
    # Generar predicciones para test
    test_predictions = pipeline.predict_test(test)
    
    # Crear submission
    submission = pipeline.create_submission(test_df, test_predictions)
    
    # Guardar submission
    submission_path = 'submission_oot.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission guardada en: {submission_path}")
    
    print("\n" + "=" * 70)
    print("FINALIZADO")
    print("=" * 70)
    
    return pipeline, submission


def compare_with_cv():
    """
    Compara OOT validation con Cross-Validation estándar.
    """
    print("\n" + "=" * 70)
    print("COMPARACIÓN: OOT vs CROSS-VALIDATION")
    print("=" * 70)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, KFold
    from time_series_cv import OutOfTimeSplit
    
    # Cargar datos
    train_df = pd.read_csv('data/train.csv')
    
    # Preparar features
    feature_cols = [col for col in train_df.columns if col not in ['id_cliente', 'mes', 'objetivo']]
    feature_cols = [col for col in feature_cols if train_df[col].dtype in [np.int64, np.float64, bool]]
    
    X = train_df[feature_cols].fillna(0)
    y = train_df['objetivo']
    
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    # 1. K-Fold estándar (CON LEAKAGE)
    print("\n1. K-FOLD ESTÁNDAR (CON LEAKAGE TEMPORAL)")
    print("-" * 70)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
    print(f"AUC CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("[X] ESTO ES LEAKAGE - mezcla futuro con pasado")
    
    # 2. Out-of-Time CV (SIN LEAKAGE)
    print("\n2. OUT-OF-TIME CV (SIN LEAKAGE)")
    print("-" * 70)
    oot_cv = OutOfTimeSplit()
    oot_scores = []
    
    for train_idx, val_idx in oot_cv.split(train_df):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, val_pred)
        oot_scores.append(score)
    
    print(f"AUC OOT CV: {np.mean(oot_scores):.4f} ± {np.std(oot_scores):.4f}")
    print("[OK] Sin leakage - respeta orden temporal")
    
    # 3. OOT Final (Train 202601-202610, Val 202611)
    print("\n3. OOT FINAL (ESTRATEGIA DE COMPETENCIA)")
    print("-" * 70)
    oot_validator = OOTValidator()
    train, val = oot_validator.split_train_val(train_df)
    
    X_train = X[X.index.isin(train.index)]
    X_val = X[X.index.isin(val.index)]
    y_train = y[y.index.isin(train.index)]
    y_val = y[y.index.isin(val.index)]
    
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    oot_final_score = roc_auc_score(y_val, val_pred)
    
    print(f"AUC OOT Final: {oot_final_score:.4f}")
    print("[OK] Simula exactamente el escenario de competencia")
    
    # Comparación
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO")
    print("=" * 70)
    print(f"K-Fold (con leakage):    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"OOT CV (sin leakage):    {np.mean(oot_scores):.4f} ± {np.std(oot_scores):.4f}")
    print(f"OOT Final (competencia): {oot_final_score:.4f}")
    diff = cv_scores.mean() - oot_final_score
    print(f"\nDiferencia K-Fold vs OOT Final: {diff:.4f}")
    print("Esta diferencia representa el overfitting por leakage temporal")
    print("=" * 70)


if __name__ == "__main__":
    # Ejemplo principal
    pipeline, submission = example_oot_validation()
    
    # Comparación con CV
    compare_with_cv()
