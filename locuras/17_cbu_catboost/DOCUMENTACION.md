# Intento 17: Clustering-Based Undersampling (CBU) con CatBoost

## Hipótesis
Si usamos clustering (K-Means) para mapear el terreno, podemos borrar estratégicamente a los "ceros" que están en clústers mixtos (zona de solapamiento) y hacer un submuestreo conservador en los clústers puros. Al devolverle los datos crudos resultantes a CatBoost, el modelo tendrá un camino limpio sin perder su poder con variables categóricas.

## Pipeline Diseñado
1. **Espacio Geométrico:** Transformamos el dataset (Target Encoder + Scaler) solo para que K-Means pueda calcular distancias.
2. **Clustering:** `MiniBatchKMeans` (50 clústers) sobre TODO el set de entrenamiento.
3. **Estrategia CBU:**
    - Si un clúster tenía más del 20% de "1s", se consideró zona invadida. Destruimos todos los "0s" de ese clúster (3,887 ceros borrados).
    - Si el clúster era pacífico (casi puros 0s), nos quedamos solo con el 20% aleatorio para no abrumar al modelo (56,767 ceros descartados).
4. **El Puente:** Filtramos el dataset *original crudo* usando los índices de los 14,172 ceros que sobrevivieron y todos los 13,254 unos.
5. **Entrenamiento:** CatBoostClassifier.

## Resultados
* **AUC en Validación:** **0.6254** (Ligeramente inferior al 0.6315 del Intento 10).
* **Recall en Clase 1:** **50%** (Inferior al 55.6% del Intento 10).

## Análisis y Conclusión Definitiva
El CBU es lógicamente impecable, pero sufre de la misma "Maldición del Espacio Euclideano" que el algoritmo ENN (Intento 14).
1. K-Means agrupó a los clientes basándose en distancias de línea recta sobre variables de texto codificadas estáticamente. 
2. Las decisiones de K-Means sobre qué es "zona pura" y qué es "zona de solapamiento" eran geométricamente correctas, pero contextualmente falsas para un modelo de lectura nativa como CatBoost.
3. **Confirmación del Techo Matemático:** Al borrar datos usando lógicas humanas o geométricas, deforma sutilmente la distribución estadística original. 

**Veredicto Final del Proyecto:** El **Random Undersampling (Intento 10)** es el Rey Absoluto. Al elegir "quién vive y quién muere" lanzando una moneda al aire (aleatorio), garantiza que el subconjunto pequeño conserve una miniatura estadísticamente perfecta del universo original. CatBoost prefiere mil veces un pequeño ecosistema caótico pero real, a uno limpiado artificialmente por otro algoritmo.
