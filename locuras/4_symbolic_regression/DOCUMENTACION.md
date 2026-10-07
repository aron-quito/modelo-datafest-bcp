# Regresión Simbólica (Mutación Genética)

## La Idea Principal
Usamos un Algoritmo Genético (librería `gplearn`). Le damos al algoritmo operaciones matemáticas básicas (`suma`, `resta`, `multiplicación`, `división`, `seno`, `logaritmo`).
El algoritmo creará 10,000 ecuaciones al azar cruzando tus variables (Ej: `Ecuacion_1 = Edad * log(Saldo)`). Evaluará cuáles se acercan más a predecir el `objetivo`. 
Las peores morirán. Las mejores se "reproducirán" cruzando sus ecuaciones entre sí para crear la siguiente generación. 
Después de 10 generaciones, extraeremos las 5 fórmulas matemáticas alienígenas más precisas y las agregaremos como columnas para CatBoost.

## Ventajas
1. **Descubrimiento No Humano:** Encuentra relaciones que a ningún analista de datos se le ocurrirían jamás.
2. **Interpretabilidad Absoluta:** A diferencia de una Red Neuronal, la Regresión Simbólica te escupe la fórmula exacta en texto plano. Puedes ver literalmente qué matemática rige a tus clientes.
3. **Expansión de Fronteras:** Si los árboles no pueden hacer matemáticas complejas (como divisiones o funciones trigonométricas en serie), nosotros les damos la respuesta ya procesada.

## Problemas que pueden ocurrir
1. **Lentitud Extrema:** Evaluar 10,000 fórmulas complejas contra 100,000 filas (1 billón de operaciones matemáticas) en cada generación hace que el CPU sufra muchísimo. Puede tomar horas si no se limita la población.
2. **Sobreajuste Salvaje (Overfitting):** El algoritmo genético es tan insistente que puede crear una ecuación gigantesca (de 50 términos) que memoriza el dataset de entrenamiento perfectamente pero falla miserablemente en el mundo real. Hay que obligarlo a penalizar ecuaciones largas (Parsimonia).
3. **División por Cero e Infinitos:** En la mutación genética aleatoria, el código frecuentemente intentará hacer `Variable / 0` o `log(-5)`, lo que crashea el script si no lo programamos con funciones "protegidas" de antemano.
