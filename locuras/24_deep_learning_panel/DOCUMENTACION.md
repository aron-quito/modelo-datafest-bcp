# Locura 24: Deep Learning con Transformers Tabulares (TabNet)

## Hipótesis
Habiendo alcanzado el límite máximo con algoritmos basados en Árboles de Decisión (CatBoost = 0.86 AUC), la última frontera de la inteligencia artificial moderna son las Redes Neuronales. Específicamente, aplicamos **TabNet**, que es una arquitectura de red neuronal profunda que utiliza un mecanismo de "Atención Secuencial" (similar a los Transformers de NLP) diseñado exclusivamente para datos tabulares.

¿Podrá la atención de una red neuronal profunda encontrar relaciones ocultas en las 49 variables de élite que CatBoost no pudo ver?

## El Experimento
1. **Preprocesamiento Estricto:** A diferencia de CatBoost, las redes neuronales exigen que todas las variables estén perfectamente estandarizadas. Escalamos las variables continuas (media 0, varianza 1) y codificamos las categóricas con tensores de "Embeddings" (una dimensión continua para las palabras).
2. **Arquitectura:** Usamos `TabNetClassifier` de PyTorch con el optimizador Adam y decaimiento de Learning Rate (`StepLR`).
3. **Poda:** Respetamos el bisturí de SHAP de la Locura 23 (solo 49 variables de élite).
4. **Validación:** 90/10 Split para detener el entrenamiento cuando la red empezara a memorizar (Early Stopping).

## Resultados de la Locura
*   **AUC TabNet (Deep Learning):** **0.8475**
*   **Época de convergencia:** 40 épocas.

## Conclusión
La red neuronal logró un impresionante **0.8475 AUC**, lo cual es un puntaje altísimo y demuestra que las 49 variables generadas en el Panel Data son matemáticamente perfectas y generalizables a cualquier algoritmo moderno.

Sin embargo, **CatBoost (0.8627 AUC) sigue siendo el rey absoluto.** 
Esto confirma la regla de oro de la ciencia de datos actual: En imágenes, texto o audio, el Deep Learning (Transformers/CNNs) es imbatible. Pero en **datos tabulares puros (Excel/CSV)** con información financiera dura, los algoritmos basados en Boosting de Árboles (Gradient Boosting) siguen dominando el estado del arte debido a su resistencia inherente al ruido y a los valores atípicos.
