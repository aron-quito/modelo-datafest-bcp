# Intento 22: Evoluciones del Panel Data (Rompiendo el 0.85)

## Hipótesis
Ahora que la base de datos está formulada correctamente (1 fila por cliente con agregaciones longitudinales), podemos aplicar todas las técnicas del estado del arte que fallaron anteriormente por culpa del solapamiento artificial, para intentar subir el AUC por encima del 0.8533.

## Las 3 Evoluciones Puestas a Prueba
Se crearon 74 variables financieras por cliente y se ejecutaron 3 pipelines en paralelo:

1. **Pipeline A: The Surgeon (SHAP Pruning + CatBoost):** 
   - De las 74 variables creadas, se usó un modelo base para calcular los Valores SHAP.
   - Se eliminaron las 25 características más ruidosas e inútiles (ej. medias de edades, días de pago preferidos, variaciones de visitas web que no sumaban nada).
   - Se entrenó un CatBoost sobre las variables de élite restantes.

2. **Pipeline B: The Balancer (SMOTE + CatBoost):**
   - Al agrupar por cliente, la clase minoritaria ahora es la de los "No Compradores (0)" (solo hay unos 8,000 ceros frente a 16,000 unos).
   - Se aplicó Target Encoding y luego SMOTE para crear 5,000 "ceros" sintéticos y nivelar el terreno.

3. **Pipeline C: The Kaggle Champion (LightGBM):**
   - Sabiendo que ahora dominan las variables puramente numéricas continuas (desviaciones estándar, tendencias), se invocó a LightGBM, el rey del cálculo numérico, hiper-parametrizado para evitar sobreajuste.

## Resultados del Torneo
1. **🏆 Pipeline A (SHAP Pruning): 0.8559 AUC**
2. **Pipeline B (SMOTE): 0.8496 AUC**
3. **Pipeline C (LightGBM): 0.8485 AUC**

## Conclusión Definitiva
**¡Nuevo y Máximo Récord: 0.8559 AUC!**

El **Pipeline A** demostró que la filosofía de "menos es más" sigue gobernando el Machine Learning. Al crear 74 variables para describir la historia del cliente, inevitablemente creamos nuevo ruido. Al usar SHAP como bisturí para extirpar las 25 métricas inútiles, CatBoost logró enfocar toda su atención en el núcleo del comportamiento financiero.

LightGBM y SMOTE no lograron superar al CatBoost limpio, confirmando que la interpolación sintética de SMOTE ensucia la realidad, y que CatBoost sigue manejando las categóricas originales mejor que el Target Encoding de LightGBM.

Este Pipeline A es la cúspide de la investigación y el modelo definitivo para la competencia Datafest.
