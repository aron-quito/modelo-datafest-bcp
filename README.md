# 🏆 BCP Datafest 2026 - Modelo de Propensión de Compra

Este repositorio documenta el proceso científico y la resolución técnica para el reto de predicción del BCP Datafest. El objetivo principal es identificar clientes con alta probabilidad de compra (Clase "1") maximizando la métrica AUC, superando la barrera del desbalanceo poblacional (85% vs 15%) y el severo solapamiento de clases.

## 📊 Matriz Comparativa de Experimentos

Durante el proyecto, iteramos desde modelos clásicos hasta "locuras" algorítmicas extremas. Descubrimos que el máximo AUC teórico rondaba el **0.6315**, pero a costa de ignorar por completo a los compradores reales (Recall = 0.0%). 

Nuestro **Modelo Campeón** sacrifica 0.001 de AUC para atrapar al **53.1%** de los compradores.

| Modelo / Arquitectura | Precisión (0) | Recall (0) | Precisión (1) | Recall (1) | F1 (1) | AUC Final |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| XGBoost (Base) | 0.85 | 1.00 | 0.00 | 0.000 | 0.00 | 0.6120 |
| LightGBM | 0.85 | 1.00 | 0.15 | 0.001 | 0.01 | 0.6250 |
| CatBoost (5to Intento) | 0.85 | 1.00 | 0.22 | 0.001 | 0.01 | **0.6315** |
| Self-Supervised Masking | 0.85 | 1.00 | 0.00 | 0.000 | 0.00 | 0.6280 |
| UMAP + Clustering | 0.85 | 1.00 | 0.00 | 0.000 | 0.00 | 0.6303 |
| PyTorch NN + SMOTETomek | 0.86 | 0.89 | 0.25 | 0.196 | 0.22 | 0.6033 |
| CatBoost + Gaussian Noise| 0.85 | 1.00 | 0.00 | 0.000 | 0.00 | 0.6293 |
| **🥇 Campeón (Tribu + Pesos)**| **0.88** | **0.64** | **0.20** | **0.531** | **0.29** | **0.6305** |

---

## 🧠 Arquitectura del Modelo Campeón

El modelo final (`campeon_catboost_tribu.cbm`) no recae en una red neuronal incomprensible, sino en un pipeline tabular robusto de 4 capas:

### 1. Capa de Ingesta de Datos Brutos
Se detectó (mediante correlación de Pearson y análisis de Tomek Links) que las variables individuales carecen de correlación lineal directa con la variable objetivo ($\rho < 0.07$). Es un espacio de datos caótico y altamente solapado.

### 2. Capa de Ingeniería de Contexto Social (La "Tribu")
Para romper el solapamiento, dejamos de mirar al cliente de forma aislada. Agrupamos a los clientes por `region` y `ocupacion`. Transformamos los datos absolutos en **relativos**:
- ¿Gana este cliente más o menos que su tribu?
- ¿Su saldo es superior a la media de su grupo?
*Esta capa dota al modelo de inteligencia socioeconómica.*

### 3. Capa de Motor de Árboles Oblivious
Se utiliza **CatBoost** configurado con árboles simétricos de profundidad 6. Es elegido por su manejo nativo (Target Encoding) de variables categóricas, evitando la explosión de dimensiones de un One-Hot Encoding clásico.

### 4. Capa de Función de Pérdida Asimétrica (Kaggle Trick)
En lugar de inventar datos sintéticos con SMOTE (que degradó nuestro AUC a 0.60 creando "clientes Frankenstein"), activamos `auto_class_weights='Balanced'`. Esto aplica una multa matemática extrema al modelo cada vez que falla al predecir a un cliente de la Clase 1, forzándolo a aprender sus patrones.

---

## 📂 Estructura del Directorio

- `/dataset/`: Datos crudos del Datafest.
- `/locuras/`: Historial de experimentos extremos (Redes Neuronales, Algoritmos Genéticos, Clustering Espacial).
- `/best_model/`: Directorio de despliegue.
  - `train_final_model.py`: Script para entrenar el modelo campeón.
  - `campeon_catboost_tribu.cbm`: Modelo binario optimizado listo para inferencia.
  - `informe_datafest.pdf`: Documento científico LaTeX con todo el rigor matemático de la exploración.

---

## 🚀 Cómo reproducir el Modelo Final

```bash
cd best_model
python train_final_model.py
```
