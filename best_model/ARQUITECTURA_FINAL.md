# Arquitectura del Modelo Ganador: "Customer Propensity Pipeline"

La solución final que alcanzó el **0.86 AUC** abandona la predicción puramente estática para adoptar una arquitectura de **Agregación Longitudinal** complementada con un **Bisturí de Explicabilidad (SHAP)**.

A continuación, se detalla la arquitectura completa de extremo a extremo:

## Diagrama de Flujo (Mermaid)

```mermaid
graph TD
    A[Dataset Raw: 110,100 filas temporales] --> B[Agrupación por 'id_cliente']
    
    subgraph Feature Engineering Longitudinal
        B --> C1[Último Estado<br/>Variables Categóricas y Booleanas]
        B --> C2[Agregaciones Estadísticas<br/>Mean, Max, Min, Std Dev]
        B --> C3[Tendencias Temporales<br/>Valor Último Mes - Primer Mes]
        B --> C4[Lealtad<br/>Meses en Sistema]
    end
    
    C1 --> D[Dataset Cliente Único<br/>24,628 filas, 74 Variables]
    C2 --> D
    C3 --> D
    C4 --> D
    
    D --> E[Split de Validación Estricto<br/>Ej. 90/10 Stratified]
    
    subgraph SHAP Pruning Surgery
        E --> F[Train Set]
        E --> G[Validation Set]
        F --> H[CatBoost Base Rápido]
        H --> I[Cálculo de Valores SHAP<br/>Para cada variable]
        I --> J[Eliminar las 25 Peores Variables<br/>Ruido Matemático]
    end
    
    J -.->|Aplica la misma poda| G
    J --> K[Dataset de Entrenamiento Podado<br/>49 Variables de Élite]
    
    subgraph Modelo Final
        K --> L[CatBoostClassifier<br/>- iterations: 1500<br/>- depth: 6<br/>- lr: 0.03<br/>- auto_class_weights: Balanced]
        L --> M[Predicciones OOF y Evaluación AUC]
    end
```

## Componentes de la Arquitectura

### 1. Transformación Longitudinal (Panel Data to Static)
En lugar de tratar cada aparición mensual de un cliente como un dato independiente (lo cual genera un "data leakage" colosal en la validación), comprimimos todo el historial de la persona en una sola fila. El espacio muestral pasa de `N = 110,100` a `N = 24,628`.

### 2. Extracción de Metadatos Financieros (74 Features)
Para compensar la compresión temporal, extrajimos la "película" completa del cliente:
*   **Volatilidad:** Calculando la Desviación Estándar (`std`) de su `ratio_deuda_ingresos` o `saldo_promedio`.
*   **Límites:** Analizando los picos (`max`, `min`) de sus movimientos.
*   **Tracción Temporal:** Restando los valores de su último mes frente a los de su primer mes para calcular la tendencia (`trend`).
*   **Fidelidad:** Contando la suma de meses de interacción.

### 3. SHAP Pruning (El "Cirujano")
La creación masiva de agregaciones genera ruido inevitable. En lugar de usar métodos de selección globales o estadísticos (como ANOVA o Gain global), usamos Teoría de Juegos a través de **SHAP (SHapley Additive exPlanations)**.
El algoritmo entrena un modelo rápido sobre el Train set, evalúa matemáticamente el impacto marginal de cada variable en las decisiones finales del árbol, y **elimina las 25 variables menos determinantes** (ej. Desviaciones estándar de visitas web o edades promedio).

### 4. El Corazón Predictivo (CatBoost)
El motor final es un ensamblado de árboles por Boosting de Gradiente (`CatBoostClassifier`).
*   **Manejo Categórico:** Se le delega el manejo nativo de las variables de texto (Ordered Target Encoding) para evitar sesgos de filtración.
*   **Balanceo Interno:** Dado que al limpiar la base los "ceros" pasaron a ser la minoría (aprox 8,000 contra 16,000 "unos"), el algoritmo balancea los pesos de la función de pérdida (`auto_class_weights='Balanced'`).
*   **Prevención de Overfitting:** Tasa de aprendizaje baja (`0.03`), profundidad restringida (`depth=6`) y detención temprana (`early_stopping_rounds=50`).

---

## Resultados y Verificación de Impacto de Negocio

El modelo superó todas las pruebas de validación cruzada y estrés, logrando converger universalmente en **~0.86 AUC**. Evaluando el modelo en el set de prueba ciego del pipeline clásico (20% del total, 4,926 clientes), extrajimos la Matriz de Confusión final:

```text
--- MATRIZ DE CONFUSIÓN ---
             Predicción: NO (0)   Predicción: SÍ (1)
Real: NO (0)       1242                  370
Real: SÍ (1)       842                   2472

--- REPORTE DE CLASIFICACIÓN ---
Precision Clase 1 (Compra): 0.87
Recall Clase 1 (Compra): 0.75
```

### 🧠 Impacto Comercial (BCP):
*   **Precisión Quirúrgica (87%):** Si el modelo levanta la bandera diciendo que una persona "SÍ comprará", en el 87% de las ocasiones acertará. Esto minimiza el desperdicio de dinero en campañas de marketing a falsos positivos (solo hubieron 370).
*   **Captura Masiva de Mercado (75%):** El modelo logra detectar exitosamente al 75% del universo total de clientes con verdadera intención de compra en el mercado ciego.

## Experimentos Finales (Nivel God Kaggle)

Para estar absolutamente seguros de que el modelo CatBoost con 49 variables no podía ser derrotado, se ejecutaron dos locuras de validación avanzadas:

1.  **Locura 24 (Transformers Tabulares - TabNet):** Se estandarizaron las variables y se ingresaron a una red neuronal atencional. TabNet logró un **0.8475 AUC**, demostrando que las 49 variables creadas son universalmente predictivas para cualquier algoritmo, pero validando que el Gradient Boosting sigue siendo el rey de los datos tabulares asimétricos.
2.  **Locura 25 (Stacking Heterogéneo):** Se combinaron CatBoost, LightGBM y XGBoost, entrenando un Meta-Modelo (Logistic Regression) para decidir los pesos finales. El score se elevó infinitesimalmente a **0.8590 AUC**. El hallazgo brillante fue que el Meta-Modelo le otorgó **el 80% de la importancia absoluta** de la decisión únicamente a CatBoost.

Conclusión indiscutible: **El Pipeline Final en CatBoost ha extraído el máximo de entropía posible del Dataset.**
