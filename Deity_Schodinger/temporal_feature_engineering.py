"""
Temporal Feature Engineering - Arquitectura Completa
====================================================

Integrado con Time-Series CV y OOT Validation.

Arquitectura:
┌────────────────────────────────────────────────────────┐
│                DATASET TEMPORAL BRUTO                  │
└───────────────────────────┬────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
 VARIABLES DEL PASADO (< t)          TRANSFORMACIONES DE GRUPOS
┌──────────────────────────────┐        ┌──────────────────────────────┐
│ • Lags (t-1, t-2)            │        │ • Target Encoding Temporal   │
│ • Deltas & Ratios            │        │   (Suavizado con Alpha)      │
│ • Rolling Windows (3m, 6m)   │        └──────────────┬───────────────┘
│ • Expanding Aggregations     │                       │
│   (media, std acumulada)     │                       │
└──────────────┬───────────────┘                       │
               │                                       │
               └───────────────────┬───────────────────┘
                                   │
                                   ▼
      ┌────────────────────────────────────────────────────────┐
      │           MATRIZ LIBRE DE DATA LEAKAGE                 │
      └───────────────────────────┬────────────────────────────┘
                                   │
           ┌──────────────────────┴──────────────────────┐
           ▼                                             ▼
 TIME-SERIES CROSS-VALIDATION                  OUT-OF-TIME (OOT) TEST
(Entrenar T, Validar T+1)                      (Caja fuerte final)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Pipeline completo de Feature Engineering Temporal.
    
    Capas:
    A. Lags & Temporal Dynamics (por id_cliente ordenado por mes)
    B. Client Aggregations (perfil histórico)
    C. Domain-Specific Features
    D. Target Encoding (expanding window por mes)
    
    Todas las transformaciones respetan el orden temporal y evitan data leakage.
    """
    
    def __init__(self, 
                 time_col: str = 'mes',
                 group_col: str = 'id_cliente',
                 target_col: str = 'objetivo',
                 lag_periods: List[int] = [1, 2, 3],
                 rolling_windows: List[int] = [3, 6],
                 alpha: float = 10.0):
        """
        Inicializar el Feature Engineer Temporal.
        
        Parámetros:
        -----------
        time_col : str
            Columna temporal (default: 'mes')
        group_col : str
            Columna de grupo para lags/aggregations (default: 'id_cliente')
        target_col : str
            Columna target para encoding (default: 'objetivo')
        lag_periods : List[int]
            Períodos de lag a calcular (default: [1, 2, 3])
        rolling_windows : List[int]
            Ventanas para rolling features (default: [3, 6])
        alpha : float
            Parámetro de suavizado para target encoding (default: 10.0)
        """
        self.time_col = time_col
        self.group_col = group_col
        self.target_col = target_col
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.alpha = alpha
        
        # Configuración de features por capa
        self.lag_features = ['saldo_promedio', 'visitas_web_ultimos_90_dias', 
                            'dias_ultima_transaccion', 'ingresos']
        self.delta_features = ['saldo_promedio', 'ingresos', 'numero_productos']
        self.rolling_features = ['saldo_promedio', 'ingresos', 'visitas_web_ultimos_90_dias']
        self.cat_features = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo']
        
        # Almacenar encoding maps
        self.encoding_maps = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Ajusta el feature engineer (calcula encoding maps).
        
        Solo calcula target encoding maps usando expanding window.
        """
        X = X.copy()
        if y is not None:
            X[self.target_col] = y
        
        # Calcular target encoding con expanding window
        self._fit_target_encoding(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas las transformaciones temporales.
        
        Orden de transformaciones:
        1. Ordenar por grupo y tiempo
        2. Calcular lags
        3. Calcular deltas y ratios
        4. Calcular rolling windows
        5. Calcular aggregations por cliente
        6. Calcular domain-specific features
        7. Aplicar target encoding
        """
        X = X.copy()
        
        # 1. Ordenar por grupo y tiempo
        X = X.sort_values([self.group_col, self.time_col])
        
        # 2. Capa A: Lags & Temporal Dynamics
        X = self._add_lags(X)
        X = self._add_deltas_and_ratios(X)
        X = self._add_rolling_windows(X)
        
        # 3. Capa B: Client Aggregations
        X = self._add_client_aggregations(X)
        
        # 4. Capa C: Domain-Specific Features
        X = self._add_domain_features(X)
        
        # 5. Capa D: Target Encoding
        X = self._apply_target_encoding(X)
        
        return X
    
    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Capa A.1: Lags (t-1, t-2, t-3)
        
        Fórmula: lag_k(x_t) = x_{t-k}
        Importancia: Captura inercia y patrones de comportamiento
        """
        for col in self.lag_features:
            if col not in df.columns:
                continue
            
            for period in self.lag_periods:
                lag_col = f"{col}_lag_{period}"
                df[lag_col] = df.groupby(self.group_col)[col].shift(period)
        
        return df
    
    def _add_deltas_and_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Capa A.2: Deltas y Ratios de Cambio
        
        Deltas: delta_t = x_t - x_{t-1}
        Ratios: ratio_t = (x_t + 1) / (x_{t-1} + 1)
        """
        for col in self.delta_features:
            if col not in df.columns:
                continue
            
            # Delta (cambio absoluto)
            delta_col = f"{col}_delta"
            lag_1 = f"{col}_lag_1"
            if lag_1 in df.columns:
                df[delta_col] = df[col] - df[lag_1]
            
            # Ratio (cambio relativo)
            ratio_col = f"{col}_ratio"
            if lag_1 in df.columns:
                df[ratio_col] = (df[col] + 1) / (df[lag_1] + 1)
        
        return df
    
    def _add_rolling_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Capa A.3: Rolling Windows (3m, 6m)
        
        Estadísticas: mean, min, max, std
        Fórmula: rolling_mean_k(t) = mean(x_{t-k+1}, ..., x_t)
        """
        for col in self.rolling_features:
            if col not in df.columns:
                continue
            
            for window in self.rolling_windows:
                # Rolling mean
                df[f"{col}_rolling_mean_{window}"] = df.groupby(self.group_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std
                df[f"{col}_rolling_std_{window}"] = df.groupby(self.group_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                )
                
                # Rolling min
                df[f"{col}_rolling_min_{window}"] = df.groupby(self.group_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
                )
                
                # Rolling max
                df[f"{col}_rolling_max_{window}"] = df.groupby(self.group_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
                )
        
        return df
    
    def _add_client_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Capa B: Agregaciones por Cliente (perfil histórico)
        
        - count_months: cuántos meses ha estado activo
        - historical_mean: media histórica
        - historical_std: volatilidad histórica
        - time_since_first: antigüedad en meses
        """
        # Count meses del cliente
        df['client_month_count'] = df.groupby(self.group_col).cumcount() + 1
        
        # Agregaciones expanding de saldo e ingresos
        for col in ['saldo_promedio', 'ingresos']:
            if col not in df.columns:
                continue
            
            # Expanding mean
            df[f"{col}_expanding_mean"] = df.groupby(self.group_col)[col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
            
            # Expanding std
            df[f"{col}_expanding_std"] = df.groupby(self.group_col)[col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).std()
            )
            
            # Expanding min/max
            df[f"{col}_expanding_min"] = df.groupby(self.group_col)[col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).min()
            )
            df[f"{col}_expanding_max"] = df.groupby(self.group_col)[col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).max()
            )
        
        # Time since first appearance
        df['time_since_first_month'] = df.groupby(self.group_col)[self.time_col].transform(
            lambda x: x - x.min()
        )
        
        return df
    
    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Capa C: Domain-Specific Features
        
        - ingreso_neto_estimado = ingresos * (1 - ratio_deuda_ingresos)
        - saturacion_productos = numero_productos / (productos_activos + 1)
        - enganche_digital = visitas_web * activo_movil
        - actividad_reciente = dias_ultima_transaccion - dias_ultima_interaccion
        """
        # Ingreso neto estimado
        if 'ingresos' in df.columns and 'ratio_deuda_ingresos' in df.columns:
            df['ingreso_neto_estimado'] = df['ingresos'] * (1 - df['ratio_deuda_ingresos'])
        
        # Saturación de productos
        if 'numero_productos' in df.columns:
            df['saturacion_productos'] = df['numero_productos'] / (df['numero_productos'] + 1)
        
        # Enganche digital
        if 'visitas_web_ultimos_90_dias' in df.columns and 'activo_movil' in df.columns:
            df['enganche_digital'] = df['visitas_web_ultimos_90_dias'] * df['activo_movil'].astype(int)
        
        # Actividad reciente
        if 'dias_ultima_transaccion' in df.columns and 'dias_ultima_interaccion' in df.columns:
            df['actividad_reciente'] = df['dias_ultima_transaccion'] - df['dias_ultima_interaccion']
        
        return df
    
    def _fit_target_encoding(self, df: pd.DataFrame):
        """
        Capa D: Target Encoding con expanding window.
        
        Fórmula suavizada: (mean * n + global_mean * alpha) / (n + alpha)
        """
        months = sorted(df[self.time_col].unique())
        
        for col in self.cat_features:
            if col not in df.columns:
                continue
            
            self.encoding_maps[col] = {}
            
            for i, month in enumerate(months):
                past_months = months[:i]
                
                if len(past_months) == 0:
                    # Primer mes: usar media global como fallback
                    past_data = df
                else:
                    past_data = df[df[self.time_col].isin(past_months)]
                
                # Calcular target mean por categoría
                target_mean = past_data.groupby(col)[self.target_col].mean()
                global_mean = past_data[self.target_col].mean()
                count = past_data.groupby(col).size()
                
                # Smooth encoding
                smoothed_encoding = (count * target_mean + self.alpha * global_mean) / (count + self.alpha)
                
                self.encoding_maps[col][month] = smoothed_encoding.to_dict()
    
    def _apply_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el target encoding pre-calculado.
        """
        for col in self.cat_features:
            if col not in df.columns or col not in self.encoding_maps:
                continue
            
            new_col = f"{col}_encoded"
            
            for month in df[self.time_col].unique():
                if month not in self.encoding_maps[col]:
                    continue
                
                encoding = self.encoding_maps[col][month]
                mask = df[self.time_col] == month
                
                df.loc[mask, new_col] = df.loc[mask, col].map(encoding).fillna(
                    list(encoding.values())[0] if encoding else 0
                )
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera un resumen de las features creadas.
        """
        original_cols = set(['id_cliente', 'mes', 'objetivo'] + 
                           self.lag_features + self.delta_features + 
                           self.rolling_features + self.cat_features)
        
        new_cols = [col for col in df.columns if col not in original_cols]
        
        summary = []
        for col in new_cols:
            if '_lag_' in col:
                layer = 'A. Lags'
            elif '_delta' in col or '_ratio' in col:
                layer = 'A. Deltas/Ratios'
            elif '_rolling_' in col:
                layer = 'A. Rolling Windows'
            elif 'client_' in col or '_expanding_' in col or 'time_since_' in col:
                layer = 'B. Client Aggregations'
            elif 'ingreso_neto' in col or 'saturacion' in col or 'enganche' in col or 'actividad_reciente' in col:
                layer = 'C. Domain-Specific'
            elif '_encoded' in col:
                layer = 'D. Target Encoding'
            else:
                layer = 'Other'
            
            summary.append({
                'feature': col,
                'layer': layer,
                'dtype': str(df[col].dtype),
                'nulls': df[col].isnull().sum(),
                'null_pct': df[col].isnull().sum() / len(df) * 100
            })
        
        return pd.DataFrame(summary)


def example_temporal_fe():
    """
    Ejemplo completo de Feature Engineering Temporal integrado con OOT.
    """
    print("=" * 70)
    print("FEATURE ENGINEERING TEMPORAL - ARQUITECTURA COMPLETA")
    print("=" * 70)
    
    # Cargar datos
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"\nTrain: {train_df.shape[0]} registros")
    print(f"Test: {test_df.shape[0]} registros")
    
    # Inicializar Feature Engineer
    fe = TemporalFeatureEngineer(
        time_col='mes',
        group_col='id_cliente',
        target_col='objetivo',
        lag_periods=[1, 2, 3],
        rolling_windows=[3, 6],
        alpha=10.0
    )
    
    # Fit en train (calcula encoding maps)
    print("\nFit en train...")
    fe.fit(train_df, train_df['objetivo'])
    
    # Transform train
    print("Transform train...")
    train_fe = fe.transform(train_df)
    print(f"Train después de FE: {train_fe.shape[0]} registros, {train_fe.shape[1]} columnas")
    
    # Transform test
    print("Transform test...")
    test_fe = fe.transform(test_df)
    print(f"Test después de FE: {test_fe.shape[0]} registros, {test_fe.shape[1]} columnas")
    
    # Resumen de features
    print("\n" + "=" * 70)
    print("RESUMEN DE FEATURES CREADAS")
    print("=" * 70)
    feature_summary = fe.get_feature_summary(train_fe)
    
    print("\nFeatures por capa:")
    print(feature_summary.groupby('layer').size())
    
    print("\nTop 20 features:")
    print(feature_summary.head(20).to_string(index=False))
    
    # Ejemplo de integración con OOT
    print("\n" + "=" * 70)
    print("INTEGRACIÓN CON OOT VALIDATION")
    print("=" * 70)
    
    from oot_validation import OOTValidator, OOTPipeline
    from sklearn.ensemble import RandomForestClassifier
    
    # Dividir OOT
    oot_validator = OOTValidator()
    train, val, test = oot_validator.print_summary(train_df, test_df)
    
    # Aplicar FE a cada split
    print("\nAplicando FE a train...")
    train_fe = fe.transform(train)
    
    print("Aplicando FE a val...")
    val_fe = fe.transform(val)
    
    print("Aplicando FE a test...")
    test_fe = fe.transform(test)
    
    # Seleccionar features numéricas (misma lista para train, val y test)
    feature_cols = [col for col in train_fe.columns 
                   if col not in ['id_cliente', 'mes', 'objetivo'] 
                   and train_fe[col].dtype in [np.float64, np.int64, bool]]
    
    # Asegurar que test tenga las mismas columnas
    missing_cols = set(feature_cols) - set(test_fe.columns)
    if missing_cols:
        print(f"\nColumnas faltantes en test: {missing_cols}")
        for col in missing_cols:
            test_fe[col] = 0
    
    print(f"\nFeatures seleccionadas para modelo: {len(feature_cols)}")
    
    # Entrenar modelo
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
    
    pipeline.fit(train_fe, val_fe)
    
    # Generar predicciones
    test_predictions = pipeline.predict_test(test_fe)
    submission = pipeline.create_submission(test_df, test_predictions)
    
    # Guardar submission
    submission_path = 'submission_temporal_fe.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission guardada en: {submission_path}")
    
    print("\n" + "=" * 70)
    print("FINALIZADO")
    print("=" * 70)
    
    return fe, train_fe, val_fe, test_fe, submission


if __name__ == "__main__":
    fe, train_fe, val_fe, test_fe, submission = example_temporal_fe()
