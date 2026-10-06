"""
Time-Series Cross-Validation Implementation
===================================================

Concepto: En datos temporales, NO puedes mezclar futuro con pasado
- Problema de K-Fold estándar: División aleatoria filtra información del futuro al pasado
- Solución: Purged Group Time-Series Split o Out-of-Time Validation
- Implementación: train hasta mes T, validar en mes T+1
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Iterator
from sklearn.model_selection import BaseCrossValidator


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Purged Group Time-Series Split para validación temporal con grupos.
    
    Este split:
    1. Mantiene el orden temporal (no mezcla futuro con pasado)
    2. Agrupa por id_cliente para evitar data leakage
    3. Elimina períodos de purga entre train y validation
    
    Parámetros:
    -----------
    n_splits : int
        Número de splits de validación
    gap : int
        Número de meses de purga entre train y validation (default: 0)
    test_size : int
        Número de meses para validación (default: 1)
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0, test_size: int = 1):
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
        
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
              groups: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera los índices para train y validation.
        
        Parámetros:
        -----------
        X : pd.DataFrame
            Features con columna 'mes' (formato AAAAMM)
        y : pd.Series
            Target variable (opcional)
        groups : pd.Series
            Identificador de grupo (id_cliente) para evitar leakage
            
        Yields:
        -------
        (train_idx, val_idx) : Tuple[np.ndarray, np.ndarray]
            Índices para entrenamiento y validación
        """
        # Obtener meses únicos ordenados
        months = sorted(X['mes'].unique())
        n_months = len(months)
        
        if n_months < self.n_splits * (self.test_size + self.gap):
            raise ValueError(
                f"No hay suficientes meses para {self.n_splits} splits. "
                f"Se necesitan al menos {self.n_splits * (self.test_size + self.gap)} meses, "
                f"pero hay {n_months}."
            )
        
        # Calcular puntos de corte para cada split
        for i in range(self.n_splits):
            # El split i usa:
            # - Train: desde el inicio hasta month_split
            # - Validation: desde month_split + gap hasta month_split + gap + test_size
            
            # Calcular el punto de corte (de atrás hacia adelante)
            max_train_end = n_months - (self.n_splits - i) * (self.test_size + self.gap)
            
            if max_train_end < 0:
                continue
                
            train_months = months[:max_train_end]
            val_start = max_train_end + self.gap
            val_months = months[val_start:val_start + self.test_size]
            
            # Obtener índices
            train_mask = X['mes'].isin(train_months)
            val_mask = X['mes'].isin(val_months)
            
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            
            yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class OutOfTimeSplit(BaseCrossValidator):
    """
    Out-of-Time Validation: Split temporal simple.
    
    Entrena hasta el mes T, valida en el mes T+1.
    Itera sobre todos los meses disponibles.
    
    Este es el método más simple y recomendado para:
    - Problemas con fuerte dependencia temporal
    - Cuando quieres simular el escenario de producción real
    """
    
    def __init__(self):
        self.n_splits = None  # Se determina dinámicamente
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
              groups: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera splits out-of-time: train hasta mes T, test en mes T+1.
        """
        months = sorted(X['mes'].unique())
        
        for i in range(1, len(months)):
            # Train: todos los meses anteriores
            train_months = months[:i]
            # Validation: mes actual
            val_months = [months[i]]
            
            train_mask = X['mes'].isin(train_months)
            val_mask = X['mes'].isin(val_months)
            
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            
            yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is not None:
            return len(X['mes'].unique()) - 1
        return None


def get_cv_summary(X: pd.DataFrame, cv_strategy: BaseCrossValidator) -> pd.DataFrame:
    """
    Genera un resumen de los splits de validación.
    
    Parámetros:
    -----------
    X : pd.DataFrame
        DataFrame con columna 'mes'
    cv_strategy : BaseCrossValidator
        Estrategia de validación
        
    Retorna:
    --------
    pd.DataFrame con resumen de splits
    """
    summary = []
    
    for i, (train_idx, val_idx) in enumerate(cv_strategy.split(X)):
        train_months = sorted(X.iloc[train_idx]['mes'].unique())
        val_months = sorted(X.iloc[val_idx]['mes'].unique())
        
        summary.append({
            'split': i + 1,
            'train_months': f"{train_months[0]} - {train_months[-1]}",
            'train_size': len(train_idx),
            'val_months': f"{val_months[0]} - {val_months[-1]}",
            'val_size': len(val_idx),
            'n_train_months': len(train_months),
            'n_val_months': len(val_months)
        })
    
    return pd.DataFrame(summary)


def example_usage():
    """
    Ejemplo de uso con los datos reales del proyecto.
    """
    print("=" * 70)
    print("EJEMPLO DE USO: Time-Series Cross-Validation")
    print("=" * 70)
    
    # Cargar datos
    df = pd.read_csv('data/train.csv')
    print(f"\nDatos cargados: {df.shape[0]} registros")
    print(f"Rango de meses: {df['mes'].min()} - {df['mes'].max()}")
    print(f"Meses únicos: {len(df['mes'].unique())}")
    
    # 1. Out-of-Time Split (método simple)
    print("\n" + "=" * 70)
    print("1. OUT-OF-TIME SPLIT (Train hasta T, Validar en T+1)")
    print("=" * 70)
    
    oot_cv = OutOfTimeSplit()
    oot_summary = get_cv_summary(df, oot_cv)
    print(oot_summary.to_string(index=False))
    
    # 2. Purged Group Time-Series Split
    print("\n" + "=" * 70)
    print("2. PURGED GROUP TIME-SERIES SPLIT")
    print("=" * 70)
    
    pgts_cv = PurgedGroupTimeSeriesSplit(n_splits=5, gap=0, test_size=1)
    pgts_summary = get_cv_summary(df, pgts_cv)
    print(pgts_summary.to_string(index=False))
    
    # 3. Ejemplo de cómo usar en un modelo
    print("\n" + "=" * 70)
    print("3. EJEMPLO DE INTEGRACIÓN CON MODELO")
    print("=" * 70)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    
    # Preparar features (simplificado)
    feature_cols = [col for col in df.columns if col not in ['id_cliente', 'mes', 'objetivo']]
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df['objetivo']
    
    print(f"\nFeatures usadas: {X.shape[1]}")
    print(f"Target balance: {y.mean():.4f}")
    
    # Usar Out-of-Time Split
    scores = []
    for fold, (train_idx, val_idx) in enumerate(oot_cv.split(df)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        val_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, val_pred)
        scores.append(score)
        
        print(f"Fold {fold + 1}: AUC = {score:.4f}")
    
    print(f"\nAUC promedio: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    print("\n" + "=" * 70)
    print("FINALIZADO")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
