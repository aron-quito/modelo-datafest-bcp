import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("Generando gráficos de solapamiento...")

# Cargar datos
df = pd.read_csv('../../dataset/train.csv')

# Para no colapsar el gráfico con 110,000 puntos (que se vería como una mancha negra),
# tomaremos una muestra aleatoria balanceada (ej: 2000 ceros y 2000 unos) para ver la frontera.
df_zeros = df[df['objetivo'] == 0].sample(2000, random_state=42)
df_ones = df[df['objetivo'] == 1].sample(2000, random_state=42)
df_sample = pd.concat([df_zeros, df_ones])

# ==========================================
# GRÁFICO 1: VARIABLES CRUDAS (Ingresos vs Saldo)
# ==========================================
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_sample, 
    x='ingresos', 
    y='saldo_promedio', 
    hue='objetivo', 
    palette={0: '#3498db', 1: '#e74c3c'}, # Azul para 0, Rojo para 1
    alpha=0.5,
    s=30
)
plt.title('Solapamiento en Variables Crudas (Ingresos vs Saldo Promedio)', fontsize=14)
plt.xlabel('Ingresos')
plt.ylabel('Saldo Promedio')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('overlap_crudo.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# GRÁFICO 2: PCA (Compresión de todas las variables a 2D)
# ==========================================
num_cols = df_sample.select_dtypes(include=[np.number]).columns.drop(['id_cliente', 'objetivo', 'mes'])
X_num = df_sample[num_cols].fillna(0)

# Estandarizamos para PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_sample['PCA_1'] = X_pca[:, 0]
df_sample['PCA_2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_sample, 
    x='PCA_1', 
    y='PCA_2', 
    hue='objetivo', 
    palette={0: '#3498db', 1: '#e74c3c'},
    alpha=0.5,
    s=30
)
plt.title('Proyección PCA 2D (Todas las variables numéricas combinadas)', fontsize=14)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('overlap_pca.png', dpi=300, bbox_inches='tight')
plt.close()

print("¡Gráficos guardados: overlap_crudo.png y overlap_pca.png!")
