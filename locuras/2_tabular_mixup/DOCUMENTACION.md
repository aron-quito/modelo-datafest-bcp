# Tabular Mixup (Clonación Mutante)

## La Idea Principal
El *Mixup* es una técnica inventada para clasificar imágenes (mezclar la foto de un perro y un gato, y decirle a la red que es 50% perro y 50% gato). 
En datos tabulares, tomamos al Cliente A (que compró, Objetivo=1) y al Cliente B (que no compró, Objetivo=0). Matemáticamente interpolamos sus datos: `Cliente C = (Cliente A * 0.4) + (Cliente B * 0.6)`. 
El objetivo del nuevo Cliente C será `0.4` (es decir, una probabilidad suave, no un 0 o 1 estricto). Alimentamos a la red neuronal con miles de estos mutantes.

## Ventajas
1. **Suavidad Matemática:** Las redes neuronales tienden a ser demasiado "agresivas" (confiadas al 100% o 0%). El Mixup las obliga a dudar y a trazar fronteras de decisión mucho más suaves y realistas.
2. **Anti-Overfitting Definitivo:** Es imposible que la red memorice el dataset porque en cada época, los clientes que ve son mutaciones aleatorias que nunca antes existieron.
3. **Aumento de Datos Infinito:** Podemos generar millones de clientes sintéticos a partir de los 100,000 originales.

## Problemas que pueden ocurrir
1. **El Problema de las Categorías:** ¿Cómo calculas el 50% de la región `west` y el 50% de la región `east`? Matemáticamente no tiene sentido. Para hacer Tabular Mixup hay que hacer One-Hot Encoding primero, pero mezclar categorías binarias genera clientes "fantasmas" que pertenecen a medias a dos regiones, lo cual ensucia la lógica del modelo.
2. **Pérdida (Loss) Especial:** No podemos usar `Focal Loss` clásica porque los objetivos ya no son 0 y 1 enteros. Hay que programar una Entropía Cruzada Suave (*Label Smoothing*).
3. **Resultados Impredecibles:** En imágenes funciona perfecto, pero en bases de datos financieras, promediar el sueldo de un joven de 18 años con el de un millonario de 60 años puede generar un perfil sintético que directamente no existe en la economía real, confundiendo a la red.
