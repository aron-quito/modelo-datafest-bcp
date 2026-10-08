# 👥 Intento 12: Ensemble de Undersampling (Comité de Expertos)

## 💡 El Concepto
En el Intento 10 borramos a 80,000 "Ceros" (No Compradores) para equilibrar la balanza con los 15,000 "Unos" (Compradores). Aunque logramos el AUC óptimo global, nos quedó la duda: *¿Y si en esos 80,000 clientes que tiramos a la basura había patrones valiosos?*

La solución es una técnica maestra llamada **Balanced Bagging** (o EasyEnsemble). En lugar de tirar datos, dividimos a los 80,000 "Ceros" en **6 grupos de 15,000**. Luego, entrenamos **6 modelos CatBoost** distintos. Cada modelo se enfrenta a todos los compradores reales, pero combate contra un grupo distinto de no compradores. 
Al final, los 6 modelos votan democráticamente si un cliente comprará o no.

## 🚀 Resultados
```text
              precision    recall  f1-score   support

           0      0.894     0.491     0.634      9353
           1      0.189     0.671     0.295      1657

⭐ AUC FINAL (Comité): 0.62959
```

## 🧠 Conclusión Científica
¡Este es el experimento más agresivo en la historia de nuestro proyecto!
Aunque el AUC global bajó ligerísimamente (de `0.6315` a `0.6295`), el **Recall de la Clase 1 explotó hasta el 67.1%**.

*   **Intento 9 (Kaggle Trick):** Atrapó al 53.1% de los compradores.
*   **Intento 10 (Undersampling Simple):** Atrapó al 55.6% de los compradores.
*   **Intento 12 (Comité de Expertos):** Atrapó al **67.1%** de los compradores.

**El Intercambio (Trade-off):** 
Para lograr este asombroso 67% de detección, el modelo se volvió un poco "paranoico". Empezó a predecir a muchos No Compradores como probables compradores (su Recall en Ceros bajó al 49%). 
Si el banco quiere vender a toda costa y no le importa enviar algunos correos de más, **¡este es el modelo de ventas perfecto!** Si el banco prefiere ser cauteloso, el Intento 10 es más equilibrado.
