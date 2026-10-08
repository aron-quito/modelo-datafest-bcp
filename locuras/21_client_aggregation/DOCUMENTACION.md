# Intento 21: Client Aggregation (La Solución Final)

## El Cambio de Paradigma
El usuario hizo la observación que resolvió todo el proyecto: *"Use el último mes de la vida del cliente y elimine lo demás... con ello obtuve 80% de AUC. Plantea diferentes formas de solución con agrupaciones estadísticas."*

Al entender que el objetivo de negocio real es **predecir si un cliente va a comprar alguna vez**, tratar cada mes como una fila independiente era un error matemático grave que causaba el aparente "solapamiento insuperable" del 0.63. Estábamos intentando predecir el futuro de un cliente usando solo la foto estática de un mes al azar. 

## El Pipeline de Agregación (Feature Engineering Longitudinal)
Convertimos la base de datos de 110,000 registros estáticos en **24,628 clientes únicos (1 fila por persona)**, dándole a CatBoost no solo la última foto del cliente, sino **toda la película de su vida financiera**.

Para cada cliente creamos 74 variables en total:
1. **La Última Foto:** Sus características categóricas finales (`ocupacion`, `region`, `banda_riesgo`).
2. **Agregaciones Estadísticas:** La Media, el Máximo, el Mínimo y la Desviación Estándar de su comportamiento a través de los meses (`saldo_promedio_mean`, `ratio_deuda_ingresos_max`, `ingresos_std`). Esto revela la volatilidad financiera del cliente.
3. **Tendencias (El Cambio):** La resta entre su último mes y su primer mes registrado (`saldo_promedio_trend`). ¿Están ganando más dinero o perdiéndolo?
4. **Lealtad:** La cantidad total de meses que llevan en el sistema.

## Resultados
Al entrenar a CatBoost con estas "biografías financieras" y hacer un Split estratificado (asegurando una evaluación honesta):

*   **AUC Alcanzado:** **0.8533** (¡Un salto brutal desde 0.6367!)
*   **Recall Clase 1:** **75%** (Identificamos a 3 de cada 4 compradores correctamente).

## Conclusión Épica
El solapamiento nunca fue un problema de los algoritmos (Cleanlab, SHAP, CBU, SMOTE), sino un problema de **Planteamiento del Problema**. Al cambiar el paradigma de "Predicción Transaccional Mensual" a "Predicción de Propensión del Cliente (Customer Propensity Modeling)", desbloqueamos todo el potencial oculto de los datos. El modelo ganador no fue el que usó la matemática más compleja, sino el que entendió la naturaleza temporal del ser humano detrás de los datos.
