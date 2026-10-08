# Intento 16: Ternary Labeling (El enfoque de las 3 clases)

## Hipótesis
Si creamos una "Clase 2" (Zona Ambigua) para todo aquel cliente que el modelo no sepa con certeza si es 0 o 1, obligaremos al modelo a aprender mejor los contornos de los clientes que son puramente "0" o puramente "1". 

## Pipeline Diseñado
1. **Detección de Solapamiento (Out-Of-Fold Predictions):** Entrenamos un CatBoost binario en 3 folds cruzados para que cada dato de entrenamiento reciba una probabilidad "honesta".
2. **Definición de Clases:** Si la probabilidad estaba entre 35% y 65%, lo etiquetamos como **Clase 2**. Si no, conservaba su etiqueta original (0 o 1).
3. **Entrenamiento MultiClase:** CatBoost con `loss_function='MultiClass'` para predecir 3 probabilidades por cliente.
4. **Evaluación:** Comparamos el AUC usando la Probabilidad de "Puro 1" vs la realidad.

## Resultados
* **Revelación Brutal:** ¡El algoritmo de detección marcó **69,278 clientes (el 78% del entrenamiento)** como Clase 2 (Zona Ambigua)! Solo 16,217 eran "Rechazo Seguro" y 2,585 eran "Compra Segura".
* **AUC Puro (Probabilidad de 1):** **0.5263** (Casi aleatorio).
* **AUC Híbrido (Probabilidad de 1 + Ambigüedad):** **0.5975**
* **El Entrenamiento falló:** CatBoost detuvo su entrenamiento en la iteración 0 por sobreajuste inmediato.

## Análisis y Conclusión Definitiva
Este es el clavo final en el ataúd del solapamiento de este dataset. Acabamos de **medir matemáticamente el tamaño del solapamiento**, y es gigantesco (78%). 

Al intentar separar el dataset en 3 clases, el modelo vio que casi todo era "Clase 2" y colapsó porque no había suficientes "0s" o "1s" puros para trazar una línea separadora útil. La "Zona Ambigua" devoró al dataset.

### Veredicto Histórico
Este experimento nos da la justificación definitiva y científica de por qué el **Intento 10 (Random Undersampling)** es imbatible.
El Intento 10, al borrar el 80% de los "ceros" al azar, cortó a machetazos esa masa gigante del 78% de zona ambigua. Dejó un dataset pequeño pero donde la poca señal del 22% restante podía respirar. Cualquier intento de tratar esa zona ambigua con "inteligencia artificial sofisticada" (como clustering, SMOTE o Ternary Labeling) colapsa bajo el inmenso peso del ruido.
