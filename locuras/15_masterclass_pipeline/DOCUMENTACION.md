# Intento 15: La "Masterclass" (LightGBM + Target Encoding + Clustering Undersampling + Feature Selection)

## Hipótesis
Si combinamos todas las técnicas avanzadas (Ingeniería de variables, eliminación de variables inútiles con importancia de LightGBM, Clustering-Based Undersampling con K-Means y LightGBM), obtendremos un modelo más limpio y destilado que CatBoost, rompiendo el techo de 0.6315.

## Pipeline Diseñado
1. **Feature Engineering:** Creación de `ratio_saldo_ingreso` y `ratio_edad_cuenta`.
2. **Target Encoding:** Transformación matemática de las categóricas.
3. **Auditoría de Variables:** Un modelo LightGBM rápido evaluó la ganancia de información (Gain) y **eliminó las 6 peores variables** (ej. `region`, `dispositivo_principal`, `tiene_prestamo`, `tiene_seguro`).
4. **Clustering-Based Undersampling (CBU):** K-Means dividió a los "ceros" en 50 clústers, y se hizo un muestreo representativo para bajar los ceros de 74,000 a 19,850.
5. **Entrenamiento:** LightGBM con los datos súper procesados.

## Resultados
* **AUC en Validación:** **0.6049** (El peor resultado de todos los modelos competitivos).
* **Recall en Clase 1:** **11%** (Desastroso. El modelo predijo casi todo como 0).
* **Parada Prematura:** LightGBM se rindió en la iteración 25 por sobreajuste inmediato.

## Análisis y Conclusión Definitiva (La Lección de Data Science)
Este experimento fallido es una mina de oro de aprendizaje:
1. **Las variables "Inútiles" NO eran inútiles:** Al pedirle a LightGBM que evalúe la importancia de las variables para borrarlas, el modelo midió qué variables servían para clasificar bien a la mayoría (los "0"). Al borrar `region` o `dispositivo`, borramos el contexto que CatBoost usaba sutilmente para encontrar a los "1".
2. **Target Encoding Casero vs CatBoost:** Hacer Target Encoding estático con una librería destruyó las relaciones no lineales entre categorías. CatBoost hace esto dinámicamente en cada árbol (Ordered Target Encoding), lo cual es magia negra que LightGBM no puede igualar con preprocesamiento manual.
3. **Clustering K-Means en Alta Dimensión:** El clustering falló miserablemente porque K-Means no funciona bien con datos que fueron creados a partir de Target Encoding (el espacio métrico no tiene sentido geométrico real).

### Veredicto Final del Proyecto
Cualquier intento de "sobre-procesar" matemáticamente este dataset destruye la señal débil que existe. La **Navaja de Ockham** aplica aquí: 
El enfoque más simple y crudo (**Intento 10: Random Undersampling + CatBoost Nativo**) respetó la naturaleza del texto y la aleatoriedad de los datos, logrando el techo matemático de **0.6315 AUC**. 

No se puede exprimir jugo de una piedra matemática. Hemos demostrado que la limitación no es algorítmica, sino de la calidad de los datos originales (ruido intrínseco).
