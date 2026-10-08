# Intento 18: La Locura de Kaggle (Cleanlab + SHAP)

Agotados todos los algoritmos clásicos, fuimos a lo más profundo del estado del arte de competencias de Kaggle buscando soluciones para "solapamiento tabular extremo".

## Hipótesis (La más loca hasta ahora)
¿Y si el solapamiento extremo no es culpa nuestra ni de los algoritmos? **¿Y si los datos nos están mintiendo?** En problemas bancarios reales, mucha gente que tiene perfil perfecto de "Comprador (1)" simplemente no vio el correo ese mes, y quedan registrados como "0". Y viceversa. Esto crea un solapamiento artificial imposible de separar.

## Ejecución
1. **Confident Learning (Cleanlab):** Usamos la librería matemática `cleanlab`, desarrollada en MIT, que cruza probabilidades out-of-fold para detectar "Label Errors" (errores de etiquetado).
2. **SHAP Values con CatBoost:** Usamos Teoría de Juegos para destripar la mente de CatBoost y ver exactamente qué columnas son oro y cuáles son basura absoluta.

## Los Descubrimientos (¡Bomba!)

### 1. El Dataset es un Mentiroso (Cleanlab)
Cleanlab ha detectado matemáticamente **9,328 clientes con etiquetas corruptas** (casi el 9% de todo el dataset).
* Hay miles de "ceros" que se comportan de forma tan idéntica a los "unos" que el algoritmo está casi seguro de que DEBERÍAN ser unos (probablemente compraron por otro canal o no se registró).
* Esto explica TODO el fracaso anterior: Le estábamos pidiendo a los algoritmos (KNN, K-Means, Redes) que aprendieran a separar gemelos idénticos.

### 2. La Basura Oculta (SHAP)
SHAP desnudó a nuestras columnas. Las que creíamos importantes son en realidad el ruido que causa el solapamiento.
* **Los Reyes del Modelo (Oro puro):** `banda_riesgo`, `numero_productos`, `dias_ultima_transaccion`.
* **La Basura Absoluta (Aportan < 0.01 de valor):** `es_nuevo_cliente`, `tiene_seguro`, `dispositivo_principal`, `ocupacion`, `tiene_prestamo`.

## Conclusión Definitiva (La "Silver Bullet")
Cualquier intento futuro de subir el AUC no pasa por inventar un modelo nuevo, sino por **limpiar la mentira de los datos**. Si eliminamos esos 9,328 clientes corruptos identificados por Cleanlab y tiramos a la basura las columnas inútiles dictadas por SHAP, el "solapamiento" se reducirá drásticamente.
