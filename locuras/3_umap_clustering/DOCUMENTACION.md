# UMAP + Clustering (Agujeros Negros Dimensionales)

## La Idea Principal
Los modelos de Machine Learning (incluso CatBoost) miran a los clientes a través de "36 dimensiones" (columnas). Es muy difícil encontrar patrones macroscópicos en tantas dimensiones.
Vamos a usar **UMAP** (Uniform Manifold Approximation and Projection) para comprimir la topología de las 36 columnas en solo 2 columnas (Eje X, Eje Y).
Una vez en 2D, usaremos un algoritmo como **K-Means** o **HDBSCAN** para detectar grupos (Tribus de clientes). El número de tribu (ej. "Tribu 12") se convertirá en una nueva columna de entrada para CatBoost.

## Ventajas
1. **Visión Holística:** Agrupa a los clientes no por cortes estrictos (como un árbol), sino por su similitud global (topológica). CatBoost entenderá de inmediato "a qué barrio" pertenece el cliente.
2. **Rápido y Efectivo:** No requiere redes neuronales masivas ni GPUs. Es pura matemática de reducción dimensional.
3. **Alta Sinergia con CatBoost:** CatBoost es el rey indiscutible manejando variables categóricas. Darle una columna nueva llamada `Cluster_ID` le permite aplicar su magia interna (*Target Encoding*) para explotar ese grupo.

## Problemas que pueden ocurrir
1. **Fuga de Datos (Data Leakage):** Si corremos UMAP sobre todo el dataset (Train + Test) al mismo tiempo, estamos espiando la distribución del Test. Hay que entrenar el UMAP *solo* en Train, y luego usarlo para transformar el Test. Sin embargo, UMAP proyectando datos nuevos no siempre es 100% estable.
2. **Ajuste de Hiperparámetros:** UMAP tiene parámetros críticos (`n_neighbors`, `min_dist`). Si elegimos mal, los 100,000 clientes se verán como una bola de ruido en 2D y el clustering no servirá de nada.
3. **Ruido Innecesario:** A veces, las "tribus" detectadas por UMAP se agrupan por cosas irrelevantes (ej. agrupa a todos los que tienen tarjeta de crédito) en lugar de agruparlos por probabilidad de churn/compra.
