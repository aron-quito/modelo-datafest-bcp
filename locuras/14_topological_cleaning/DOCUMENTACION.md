# Intento 14: Limpieza Topológica con Edited Nearest Neighbors (ENN)

## Hipótesis
Si el bajo AUC se debe al solapamiento extremo entre las clases 0 y 1, usar una limpieza geométrica (Edited Nearest Neighbors - ENN) para borrar a los "0" que están en "territorio enemigo" (rodeados de "1"s) limpiará la frontera de decisión y mejorará el AUC, superando el Random Undersampling del intento 10.

## Pipeline Diseñado
1. **Separación:** Train / Validation (80/20).
2. **Preprocesamiento Temporal:** `TargetEncoder` para categóricas y `StandardScaler` para numéricas. Esto crea un espacio euclidiano para calcular distancias.
3. **Limpieza Topológica (ENN):** Eliminó ~19,455 clientes "0" de la nube de solapamiento.
4. **Puente de Índices:** Recuperamos las filas crudas (con categóricas nativas) correspondientes a los sobrevivientes.
5. **Entrenamiento:** CatBoost con `auto_class_weights='Balanced'` sobre este subconjunto de datos limpio.

## Resultados
* **AUC en Validación:** 0.6273 (Menor que el 0.6315 del Intento 10).
* **Recall en Clase 1:** 54% (Menor que el 55.6% del Intento 10).

## Análisis y Conclusión Definitiva
1. **La Maldición de la Dimensión Categórica:** El ENN se basa en distancia Euclidiana. Al forzar variables cualitativas (`profesion`, `dispositivo`) a ser números con *Target Encoding*, creamos un espacio geométrico artificial. ENN borró 19,455 ejemplos asumiendo que eran ruido, pero para CatBoost (que lee texto puro) esos ejemplos sí contenían información estructural valiosa.
2. **El Techo es de Cristal:** Hemos agotado las vías de Data Augmentation, Ensembles Balanceados (EasyEnsemble) y ahora Limpieza Geométrica (ENN). En todos los casos sofisticados, el modelo empeora ligeramente frente al **Random Undersampling Clásico (Intento 10)**.
3. **Veredicto:** El modelo "Best Model" sigue siendo el del Intento 10. La aletoriedad del Undersampling resulta ser más robusta para preservar la distribución de probabilidad marginal de las categóricas que los algoritmos de k-vecinos.
