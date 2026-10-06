import pandas as pd
import numpy as np

def aplicar_ingenieria_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Toma el DataFrame crudo y genera relaciones matemáticas complejas
    que los árboles de decisión no pueden descubrir por sí mismos.
    """
    print("🧪 Iniciando Feature Engineering Extremo...")
    df_new = df.copy()
    
    # ---------------------------------------------------------
    # 1. INGENIERÍA FINANCIERA (Ratios y Descomposiciones)
    # ---------------------------------------------------------
    # Recuperar la deuda real (absoluta) que estaba oculta en el ratio
    df_new['deuda_absoluta'] = df_new['ingresos'] * df_new['ratio_deuda_ingresos']
    
    # Capacidad de ahorro / Liquidez aparente
    # (Cuánto de sus ingresos anuales representa su saldo promedio)
    df_new['ratio_saldo_ingresos'] = df_new['saldo_promedio'] / (df_new['ingresos'] + 1)
    
    # Ingreso per cápita por edad (Métrica de "Éxito Financiero Relativo")
    df_new['ingreso_por_edad'] = df_new['ingresos'] / (df_new['edad'] + 1)
    
    # Saldo por cada producto que tiene (Ticket promedio por producto)
    df_new['saldo_por_producto'] = df_new['saldo_promedio'] / (df_new['numero_productos'] + 0.1)
    
    # ---------------------------------------------------------
    # 2. INGENIERÍA DE COMPORTAMIENTO TEMPORAL
    # ---------------------------------------------------------
    # ¿Interactúa pero no transacciona? (Lag de inactividad financiera)
    df_new['brecha_interaccion_transaccion'] = df_new['dias_ultima_interaccion'] - df_new['dias_ultima_transaccion']
    
    # Velocidad de adquisición de productos (Productos por año de antigüedad)
    df_new['velocidad_productos'] = df_new['numero_productos'] / (df_new['antiguedad_cuenta_meses'] / 12 + 0.1)
    
    # Frecuencia de visitas web (visitas por día en los últimos 90)
    df_new['frecuencia_visitas_diarias'] = df_new['visitas_web_ultimos_90_dias'] / 90.0
    
    # ---------------------------------------------------------
    # 3. INTERACCIONES LOGÍSTICAS Y DE VIDA
    # ---------------------------------------------------------
    # ¿Se mudó recientemente respecto a la apertura de la cuenta?
    df_new['mudanza_vs_cuenta'] = df_new['antiguedad_direccion_meses'] - df_new['antiguedad_cuenta_meses']
    
    # Agrupación de edad (Generaciones financieras)
    df_new['generacion'] = pd.cut(df_new['edad'], bins=[0, 25, 40, 55, 100], labels=['GenZ', 'Millennial', 'GenX', 'Boomer'])
    
    # ---------------------------------------------------------
    # 4. PRODUCTOS BANCARIOS OCULTOS
    # ---------------------------------------------------------
    # Suma de productos booleanos conocidos
    df_new['productos_conocidos'] = df_new['tiene_tarjeta_credito'].astype(int) + df_new['tiene_prestamo'].astype(int) + df_new['tiene_seguro'].astype(int)
    
    # Productos misteriosos (Cuentas de ahorro corrientes u otros que no están en las booleanas)
    df_new['productos_desconocidos'] = df_new['numero_productos'] - df_new['productos_conocidos']
    
    # ---------------------------------------------------------
    # 5. CRUCES CATEGÓRICOS PARA EL ÁRBOL
    # ---------------------------------------------------------
    # Forzar al árbol a ver la interacción explícita entre Ocupación y Región
    df_new['perfil_socioeconomico'] = df_new['ocupacion'].astype(str) + "_" + df_new['region'].astype(str)
    
    # Forzar interacción entre Riesgo y Canal
    df_new['perfil_riesgo_canal'] = df_new['banda_riesgo'].astype(str) + "_" + df_new['canal_adquisicion'].astype(str)

    print(f"✅ Feature Engineering completado. Columnas originales: {df.shape[1]}, Nuevas columnas: {df_new.shape[1]}")
    return df_new

if __name__ == "__main__":
    df = pd.read_csv('../../dataset/train.csv')
    df_eng = aplicar_ingenieria_variables(df)
    print(df_eng.head())
