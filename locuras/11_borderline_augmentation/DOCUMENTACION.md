# 🧬 Intento 11: Data Augmentation Quirúrgica (Borderline-SMOTE)

## 💡 El Problema (Por qué fallaron las otras aumentaciones)
En nuestros intentos anteriores de Data Augmentation fracasamos por dos razones:
1. **Ruido Gaussiano:** Al mover a los compradores sutilmente en todas direcciones, empujamos a muchos de ellos al territorio de los No Compradores, empeorando el solapamiento.
2. **SMOTE Clásico (Red Neuronal):** Interpoló a lo ciego. Unió compradores que estaban en extremos opuestos del mapa, trazando líneas que cruzaban territorio enemigo y creando "Clientes Frankenstein" imposibles.

## 🧠 La Nueva Reformulación: Borderline-SMOTE
Para este intento, reformularemos la técnica usando topología avanzada:
1. Escanearemos a cada cliente de la Clase 1 usando *K-Nearest Neighbors*.
2. Ignoraremos a los compradores "seguros" (rodeados de otros compradores).
3. Ignoraremos a los compradores "ruido" (solitarios rodeados 100% por no compradores).
4. **Objetivo Quirúrgico:** Nos centraremos EXCLUSIVAMENTE en los compradores que están en la "zona de guerra" (la frontera exacta donde se tocan los 0s y los 1s). 
5. Interpolaremos clones sintéticos a lo largo de esa frontera para construir un muro matemático denso de Clase 1, impidiendo que el árbol de decisión se pase hacia el lado equivocado.

## 🛠️ Ejecución
Ejecutar el script `run_borderline.py` que aplicará Target Encoding temporal a las variables categóricas, ejecutará las matemáticas espaciales de Borderline-SMOTE, y entrenará un CatBoost evaluado contra una muestra real no contaminada.
