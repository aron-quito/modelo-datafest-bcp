# Self-Supervised Masking (Pre-entrenamiento Tabular)

## La Idea Principal
Inspirado en cómo se entrenan modelos como BERT o ChatGPT. En lugar de pedirle a la Red Neuronal que adivine si el cliente comprará o no (el `objetivo`), tomamos los datos de los clientes y **borramos al azar el 30% de sus variables** (ponemos ceros o ruido). 
El trabajo de la red será reconstruir y adivinar cuáles eran esos valores originales (ej. adivinar el `saldo_promedio` basándose en la `edad` y la `ocupacion`).
Una vez que la red es experta en la "física" de los clientes del banco, congelamos su cerebro, le quitamos la capa final, y le ponemos una capa nueva que prediga el `objetivo`.

## Ventajas
1. **Entendimiento Profundo:** La red aprende la correlación oculta entre todas las variables sin sesgarse por el `objetivo`.
2. **Representación Latente:** Convierte las 30+ variables en un vector comprimido (Latent Space) de altísima calidad matemática.
3. **Inmunidad al Ruido:** Como fue entrenada para adivinar datos faltantes, se vuelve extremadamente resistente a clientes con datos anómalos o ruidosos.

## Problemas que pueden ocurrir
1. **Altísimo Costo Computacional:** Requiere entrenar la red durante cientos de épocas solo para la fase de reconstrucción, y luego otro entrenamiento para la clasificación.
2. **No garantiza relevancia:** Puede que la red aprenda perfectamente a predecir el `saldo` a partir de la `edad`, pero que esa relación no sirva absolutamente de nada para predecir si el cliente hará *Churn* o comprará el producto final (el `objetivo`).
3. **Escalado Crítico:** Si las variables no están perfectamente estandarizadas, la red intentará optimizar el error de las columnas con números más grandes (ej. `ingresos`) e ignorará las pequeñas (ej. `numero_productos`).
