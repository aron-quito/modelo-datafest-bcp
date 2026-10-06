# Arquitectura Temporal ML - Datafest BCP

Implementación de Machine Learning para datos temporales con:
- Time-Series Cross-Validation
- Feature Engineering Temporal
- OOT Validation
- Modelos GBDT (XGBoost, LightGBM, CatBoost)

---

## Resultados (AUC / Gini)

| Modelo | AUC | Gini | Archivo |
|--------|-----|------|---------|
| XGBoost Simple | 0.5609 | 0.1218 | `xgboost_simple.py` |
| Stacking 3 Modelos | 0.5676 | 0.1352 | `stacking_simple.py` |
| **CatBoost + FE** | **0.6229** | **0.2458** | `catboost_fe.py` ⭐ |

**Recomendación:** Usar `catboost_fe.py` para el mejor resultado.

---

## Arquitectura

```
DATASET → Feature Engineering Temporal → OOT Validation → Modelos → Submission
```

### Componentes:

1. **Time-Series Cross-Validation** (`time_series_cv.py`)
   - Out-of-Time Split (OOT)
   - Purged Group Time-Series Split (PGTSS)
   - Evita data leakage temporal

2. **OOT Validation** (`oot_validation.py`)
   - Train (202601-202610) → Val (202611) → Test (202612)
   - Simula escenario de competencia

3. **Feature Engineering Temporal** (`temporal_feature_engineering.py`)
   - Lags (t-1, t-2, t-3)
   - Rolling Windows (3m, 6m)
   - Aggregations por cliente
   - Target Encoding con expanding window
   - Features domain-specific

---

## Scripts para Ejecutar

### 1. XGBoost Simple (Baseline Rápido)

**Archivo:** `xgboost_simple.py`

**Características:**
- Features numéricas básicas (17 features)
- OOT Validation
- Rápido (~30 segundos)
- AUC: 0.5609 / Gini: 0.1218

**Ejecutar:**
```bash
cd deity_Schodinger
python xgboost_simple.py
```

**Archivos generados:**
- `submission_xgboost_simple.csv`

**Requisitos:**
```bash
pip install xgboost
```

---

### 2. Stacking 3 Modelos (XGBoost + LightGBM + CatBoost)

**Archivo:** `stacking_simple.py`

**Características:**
- XGBoost, LightGBM, CatBoost
- Features numéricas básicas
- Meta-modelo LogisticRegression
- OOT Validation
- AUC: 0.5676 / Gini: 0.1352

**Ejecutar:**
```bash
cd deity_Schodinger
python stacking_simple.py
```

**Archivos generados:**
- `submission_stacking_simple.csv`

**Requisitos:**
```bash
pip install xgboost lightgbm catboost
```

---

### 3. CatBoost + Feature Engineering ⭐ MEJOR RESULTADO

**Archivo:** `catboost_fe.py` ✅ **RECOMENDADO**

**Características:**
- CatBoost (mejor modelo individual)
- Feature Engineering Temporal completo
- Lags, rolling windows, aggregations
- Target encoding con expanding window
- **AUC: 0.6229 / Gini: 0.2458** - ¡Objetivo >0.6 alcanzado!

**Ejecutar:**
```bash
cd deity_Schodinger
python catboost_fe.py
```

**Archivos generados:**
- `submission_catboost_fe.csv`

**Requisitos:**
```bash
pip install catboost
```

**Mejora vs baseline:** +0.0620 AUC (+0.1240 Gini)

---

## Archivos del Proyecto

### Infraestructura (No ejecutar directamente):
- `time_series_cv.py` - Time-Series Cross-Validation
- `oot_validation.py` - Out-of-Time Validation
- `temporal_feature_engineering.py` - Feature Engineering Temporal

### Modelos (Ejecutar estos):
- `xgboost_simple.py` - XGBoost baseline
- `stacking_simple.py` - Stacking 3 modelos
- `catboost_fe.py` - CatBoost con FE ⭐ Mejor resultado

---

## Flujo Recomendado

1. **Submit rápido:** `python xgboost_simple.py` (30s, AUC 0.5609)
2. **Submit mejor:** `python catboost_fe.py` (5m, AUC 0.6229) ⭐
3. **Experimentar:** `python stacking_simple.py` (2m, AUC 0.5676)

---

## Requisitos

### Mínimo:
```bash
pip install pandas numpy scikit-learn xgboost
```

### Completo:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost
```

---

## Concepto Clave

En datos temporales, **NO puedes mezclar futuro con pasado**. 

- ❌ K-Fold estándar: Mezcla futuro con pasado (data leakage)
- ✅ OOT Validation: Respeta orden cronológico
- ✅ Feature Engineering: Solo usa información del pasado (expanding window)

**Regla de oro:** Cada fila solo puede usar información de meses ≤ su mes actual.
