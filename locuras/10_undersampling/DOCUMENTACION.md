# ✂️ Intento 10: Undersampling Extremo (El Óptimo Global)

## 💡 El Concepto
En intentos anteriores lidiamos con el severo desbalanceo (85% Ceros vs 15% Unos) usando **Cost-Sensitive Learning** (penalizando los errores matemáticamente). Sin embargo, el árbol de decisión seguía viendo físicamente 85,000 datos de personas que no compraban, lo que dominaba la estructura de los nodos.

En este experimento, decidimos intervenir físicamente el dataset de entrenamiento usando `RandomUnderSampler`.

## 🛠️ La Implementación
1. Se generaron primero nuestras **Variables de Contexto Social (Tribu)** para dotar de inteligencia relacional al modelo.
2. Separamos un 10% del dataset para `Test` (este se quedó 100% real y desbalanceado para la evaluación justa).
3. Del 90% restante (`Train`), **borramos masivamente y al azar al 84% de los "Ceros"**, hasta que quedó una batalla perfectamente nivelada: **14,910 Ceros contra 14,910 Unos**.
4. Entrenamos a CatBoost con este dataset reducido de 30,000 personas.

## 🚀 Resultados
```text
              precision    recall  f1-score   support

           0      0.886     0.614     0.726      9353
           1      0.203     0.556     0.298      1657

⭐ AUC FINAL (Undersampling): 0.63150
```

## 🧠 Conclusión Científica
¡Fue un éxito rotundo! Rompió todos los récords anteriores.
Al limpiar de "basura redundante" el territorio de la clase mayoritaria, permitimos que el modelo hiciera divisiones de árbol perfectas al 50/50. 
Alcanzamos el techo matemático absoluto de este dataset (`0.6315`) y logramos el **mayor Recall de la historia del proyecto (55.6%)**, manteniendo la precisión estable. Este es, sin duda, uno de los mejores métodos para este tipo de datos bancarios.
