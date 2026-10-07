import pandas as pd
import numpy as np

print("==================================================")
print("🔍 ANÁLISIS FORENSE DEL DATASET")
print("==================================================")

df = pd.read_csv('../../dataset/train.csv')

print("\\n1. ANÁLISIS DE VALORES NULOS (MISTERIOS OCULTOS)")
null_counts = df.isnull().sum()
null_cols = null_counts[null_counts > 0]
if len(null_cols) > 0:
    for col, count in null_cols.items():
        print(f"  - {col}: {count} nulos ({count/len(df)*100:.1f}%)")
else:
    print("  - ¡No hay valores nulos detectados!")

print("\\n2. ANÁLISIS DE CARDINALIDAD CATEGÓRICA")
cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
for col in cat_cols:
    uniques = df[col].nunique()
    print(f"  - {col}: {uniques} valores únicos.")
    if uniques > 50:
        print(f"    ⚠️ ALERTA: Alta cardinalidad. Podría estar causando ruido.")

print("\\n3. ANÁLISIS DE EXTREMOS Y OUTLIERS (NUMÉRICOS)")
num_cols = df.select_dtypes(include=[np.number]).columns.drop(['id_cliente', 'objetivo'], errors='ignore')
for col in num_cols:
    q1 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    min_val = df[col].min()
    max_val = df[col].max()
    
    # Check for extreme crazy max values (like 99999999 or -999 which might be error codes)
    if max_val > (q99 * 5) and max_val > 0:
        print(f"  - ⚠️ OUTLIER en {col}: Max {max_val} es excesivamente mayor que el P99 ({q99}). ¿Código de error oculto?")
    if min_val < 0 and q1 >= 0:
        print(f"  - ⚠️ VALOR NEGATIVO RARO en {col}: Min {min_val}. ¿Errores de captura?")

print("\\n4. ANÁLISIS DE CORRELACIÓN LINEAL CON EL OBJETIVO")
corrs = df[num_cols].corrwith(df['objetivo']).abs().sort_values(ascending=False)
print("  Top 5 Variables más correlacionadas linealmente:")
print(corrs.head(5))
print("\\n  ⚠️ Si la correlación lineal máxima es muy baja (ej. < 0.1), el problema es NO-LINEAL y caótico.")

print("==================================================")
