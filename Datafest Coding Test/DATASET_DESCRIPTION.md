# Descripción del conjunto de datos

Esta competencia utiliza un conjunto de datos para predecir la propensión de conversión de los clientes. Cada fila representa a un cliente observado durante un mes específico. Un mismo cliente puede aparecer en varios meses.

El objetivo es estimar la probabilidad de conversión del cliente. Los datos de entrenamiento abarcan de enero a noviembre de 2026, mientras que los datos de prueba corresponden a diciembre de 2026. 

## Archivos

- **train.csv**: contiene 110.100 observaciones de entrenamiento y 25 columnas. Incluye todas las variables predictoras y la columna objetivo, `objetivo`.
- **test.csv**: contiene 9.900 observaciones de diciembre y 24 columnas. Incluye las mismas variables predictoras que `train.csv`, pero no contiene `objetivo`.
- **sample_submission.csv**: muestra la estructura requerida para la entrega e incluye una fila por cada observación de `test.csv`.
- **metaData.csv**: contiene las descripciones, tipos de datos y notas de las columnas del conjunto de datos y del archivo de entrega.

## Formato de los datos

- Los archivos utilizan valores separados por comas e incluyen una fila de encabezado.
- `id_cliente` y `mes` identifican cada observación cliente-mes.
- `mes` utiliza el formato entero `AAAAMM` (por ejemplo, `202601`).
- Las columnas booleanas se representan mediante `True` y `False`.
- Los valores numéricos decimales utilizan el punto como separador decimal.
- Los archivos proporcionados para la competencia no contienen valores faltantes.

## Columnas

### Identificadores

- `id_cliente`: identificador único del cliente. Un cliente puede aparecer en más de un mes.

### Variables numéricas

- `edad`: edad del cliente.
- `ingresos`: ingresos anuales.
- `ratio_deuda_ingresos`: proporción de deuda sobre ingresos.
- `antiguedad_cuenta_meses`: antigüedad de la cuenta en meses.
- `numero_productos`: número de productos bancarios activos.
- `saldo_promedio`: saldo promedio de la cuenta.
- `dias_ultima_transaccion`: días transcurridos desde la transacción más reciente.
- `antiguedad_direccion_meses`: tiempo en meses asociado a la dirección registrada.
- `visitas_web_ultimos_90_dias`: número de visitas al sitio web durante los últimos 90 días.
- `distancia_sucursal_km`: distancia aproximada a la sucursal más cercana, en kilómetros.
- `dia_preferido_pago`: día preferido del mes para realizar pagos.
- `dias_ultima_interaccion`: días transcurridos desde la interacción más reciente registrada con el cliente.

### Variables categóricas

- `ocupacion`: categoría laboral del cliente.
- `region`: región del cliente.
- `canal_adquisicion`: canal mediante el cual fue adquirido el cliente.
- `banda_riesgo`: segmento de riesgo del cliente.
- `dispositivo_principal`: dispositivo principal registrado.

### Variables booleanas

- `tiene_tarjeta_credito`: indica si el cliente tiene una tarjeta de crédito.
- `activo_movil`: indica si el cliente está activo en la aplicación móvil.
- `es_nuevo_cliente`: indica si el cliente está identificado como nuevo.
- `tiene_prestamo`: indica si el cliente tiene un préstamo activo.
- `tiene_seguro`: indica si el cliente tiene un producto de seguro.

### Variable objetivo

- `objetivo`: indicador binario de conversión disponible únicamente en `train.csv`. Un valor de `1` representa la primera conversión del cliente durante ese mes; un valor de `0` representa que no hubo conversión.

## Evaluación

Las entregas se evalúan mediante el coeficiente de Gini derivado del ROC AUC:

`Gini = 2 * AUC - 1`

Un Gini más alto indica una mejor capacidad para ordenar a los clientes según su propensión de conversión.

## Formato de entrega

La entrega debe incluir una predicción para cada fila de `test.csv`, conservando los valores originales de `id_cliente` y el orden de las filas. Aunque `mes` forma parte del conjunto de datos, no debe incluirse en el archivo de entrega porque todas las observaciones de prueba corresponden al mismo mes.

El archivo debe contener exactamente las siguientes columnas:

```text
id_cliente,prediccion
```

`prediccion` debe ser una probabilidad entre `0` y `1`. Los valores incluidos en `sample_submission.csv` son únicamente ejemplos y deben sustituirse por las predicciones del modelo.

## Terminología

- **AUC**: área bajo la curva ROC.
- **Gini**: métrica de ordenamiento calculada como `2 * AUC - 1`.
- **AAAAMM**: año de cuatro dígitos seguido por el mes de dos dígitos.
