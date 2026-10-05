Created diagnose_model.py
Ran command: `.venv/bin/python diagnose_model.py`
Used tool: manage_task
Ran command: `sleep 15 && tail -n 25 /Users/aron/.gemini/antigravity-ide/brain/f04da4b0-3628-4d73-9971-726396337c5b/.system_generated/tasks/task-144.log`

Se ejecutó el script de diagnóstico profundo [diagnose_model.py](file:///Users/aron/githubRepo/modelo-datafest-bcp/diagnose_model.py) y se generaron los gráficos [feature_importance_lightgbm.png](file:///Users/aron/githubRepo/modelo-datafest-bcp/feature_importance_lightgbm.png) y [correlation_lgb_catboost.png](file:///Users/aron/githubRepo/modelo-datafest-bcp/correlation_lgb_catboost.png).

A continuación tienes el informe detallado de los 3 objetivos evaluados:

---

### 1. Feature Importance (LightGBM por Ganancia / *Gain*)

#### 🔝 Las 5 Variables Más Dominantes:
| Variable | % Ganancia (Gain) | Rol en el Modelo |
| :--- | :---: | :--- |
| **`banda_riesgo`** | **15.46%** | Segmento principal de partición del árbol. |
| **`numero_productos`** | **10.11%** | Nivel de vinculación del cliente con el banco. |
| **`dias_ultima_transaccion`** | **9.82%** | Recencia transaccional. |
| **`ocupacion_region`** | **6.43%** | Cruce de localización y tipo de empleo. |
| **`canal_banda`** | **4.83%** | Interacción entre canal de adquisición y riesgo. |

#### 🔻 Las 10 Variables Más Ignoradas (Menor Ganancia):
1. `ingresos_por_encima_region` (0.00%)
2. `saldo_por_encima_region` (0.00%)
3. `ingresos_por_encima_ocup` (0.00%)
4. `ocupacion` directa (0.003% - absorbida por `ocupacion_region`)
5. `region` directa (0.009% - absorbida por `ocupacion_region`)
6. `saldo_por_encima_ocup` (0.01%)
7. `grp_ocup_ratio_deuda_ingresos_median` (0.02%)
8. `grp_region_ingresos_median` (0.03%)
9. `grp_ocup_saldo_promedio_median` (0.03%)
10. `dispositivo_principal` (0.04%)

> 📌 **Conclusión 1:** Los flags binarios manuales (*"está por encima del promedio"*) aportan 0 ganancia porque los árboles de decisión ya calculan sus propios cortes numéricos sobre los ratios continuos. Es mejor mantener los ratios continuos y eliminar los flags booleanos redundantes.

---

### 2. Análisis de Errores: Clientes "1" Incomprendidos (Falsos Negativos Severos)

Se compararon los **3,314 clientes reales que sí convirtieron** a los que el modelo les asignó una probabilidad bajísima ($\le 0.134$) frente a los **3,314 convertidores bien detectados** ($\ge 0.209$):

#### 🚨 El Gran Sesgo del Modelo:
```
Distribución de banda_riesgo:
  - En Aciertos (Bien Detectados):     98.9% son 'low'  |  1.0% 'medium'  |  0.1% 'high'
  - En Incomprendidos (Falsos Neg.):   31.1% son 'low'  | 37.2% 'medium'  | 31.7% 'high'
```

#### 📊 Perfil de los Clientes que Estamos Fallando:
1. **Fuerte penalización por Riesgo:** El modelo aprendió una regla casi categórica: *"si no eres riesgo 'low', casi nunca conviertes"*. Por ello, los clientes de riesgo `medium` y `high` que sí se animan a convertir quedan completamente descartados.
2. **Inactividad y Poca Vinculación:**
   * **Productos:** Los incomprendidos tienen solo **1.66 productos** (vs 3.14 en aciertos, una diferencia del **-47.2%**).
   * **Días sin transacciones:** Los incomprendidos tienen en promedio **233.8 días** sin operar (vs 166.5 días en aciertos, **+40.4% de inactividad**).
3. **Menor adopción digital:** Solo el **65.0%** usa la app móvil (vs **85.5%** en los clientes bien detectados).

---

### 3. Matriz de Correlación de Predicciones y Errores (LightGBM vs CatBoost)

* **Correlación de Predicciones (Pearson):** **`90.36%`**
* **Correlación de Predicciones por Rangos (Spearman):** **`82.76%`**
* **Correlación de Errores Residuales $(y - \hat{y})$:** **`99.70%`**

> ⚠️ **Diagnóstico del Ensamblaje:** 
> La correlación de errores es prácticamente del **`99.7%`**. Esto significa que LightGBM y CatBoost **se equivocan exactamente en los mismos clientes**. El promedio simple entre ambos aporta muy poca ganancia porque ambos modelos sufren del mismo sesgo estructural respecto a `banda_riesgo` y `numero_productos`.

---

### 💡 3 Acciones Concretas para Romper este Techo de AUC:

1. **Regularizar el peso de `banda_riesgo` o crear Target Encoding Suavizado (Smoothed Target Encoding):** Para evitar que el árbol descarte masivamente a los clientes `medium` y `high`.
2. **Features específicas para clientes de "baja vinculación / riesgo medio":** Ratios como `visitas_web / (numero_productos + 1)` o `saldo_promedio / (dias_inactividad_relativa + 1)` para detectar interés digital en clientes poco vinculados.
3. **Incorporar un modelo de arquitectura distinta (Diversidad Real):** Una **Red Neuronal Tabular (MLP/TabNet)** o una **Regresión Logística Regularizada (Ridge/Lasso)** para el ensamble. Al tener funciones de pérdida lineales/suaves, no cometerán los mismos errores de corte rígido que los árboles.

¿Te gustaría que implementemos alguna de estas soluciones en [LightGBM.ipynb](file:///Users/aron/githubRepo/modelo-datafest-bcp/LightGBM.ipynb) (por ejemplo, el *Smoothed Target Encoding* en `banda_riesgo`, nuevas variables para clientes de riesgo medio/alto, o un ensamble con un modelo lineal regularizado para diversificar las predicciones)?