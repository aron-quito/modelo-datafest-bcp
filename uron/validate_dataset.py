import pandas as pd
import numpy as np
import os

def check_dataset():
    print("=" * 60)
    print("🔍 REVISIÓN DE INTEGRIDAD Y CALIDAD DEL DATASET")
    print("=" * 60)

    # 1. Carga de archivos
    files = {
        'train': 'dataset/train.csv',
        'test': 'dataset/test.csv',
        'sub': 'dataset/sample_submission.csv',
        'meta': 'dataset/metaData.csv'
    }

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"❌ Error: Archivo faltante '{path}'")
            return
    
    train = pd.read_csv(files['train'])
    test = pd.read_csv(files['test'])
    sub = pd.read_csv(files['sub'])
    meta = pd.read_csv(files['meta'])

    print(f"✅ Archivos cargados correctamente.")
    print(f"   - Train shape: {train.shape[0]:,} filas x {train.shape[1]} columnas")
    print(f"   - Test shape:  {test.shape[0]:,} filas x {test.shape[1]} columnas")
    print(f"   - Sample sub:  {sub.shape[0]:,} filas x {sub.shape[1]} columnas\n")

    # 2. Validación de columnas y dimensiones
    print("--- 1. Validación de Columnas y Estructura ---")
    expected_train_cols = set(train.columns)
    expected_test_cols = set(test.columns)
    
    diff_test_train = expected_train_cols - expected_test_cols
    diff_train_test = expected_test_cols - expected_train_cols

    if diff_test_train == {'objetivo'} and len(diff_train_test) == 0:
        print("✅ Las columnas de 'train' y 'test' coinciden exactamente (con 'objetivo' solo en train).")
    else:
        print(f"⚠️ Discrepancia en columnas:")
        print(f"   En train pero no en test: {diff_test_train}")
        print(f"   En test pero no en train: {diff_train_test}")

    # 3. Valores nulos
    print("\n--- 2. Valores Faltantes (Nulls / NaNs) ---")
    train_nulls = train.isnull().sum().sum()
    test_nulls = test.isnull().sum().sum()
    if train_nulls == 0 and test_nulls == 0:
        print("✅ No hay valores nulos en 'train' ni en 'test' (0 NaNs).")
    else:
        print(f"⚠️ Se encontraron nulos: Train ({train_nulls}), Test ({test_nulls})")

    # 4. Duplicados
    print("\n--- 3. Verificación de Duplicados ---")
    train_dup = train.duplicated(subset=['id_cliente', 'mes']).sum()
    test_dup = test.duplicated(subset=['id_cliente', 'mes']).sum()
    if train_dup == 0 and test_dup == 0:
        print("✅ No existen registros duplicados por (id_cliente, mes).")
    else:
        print(f"⚠️ Duplicados por clave (id_cliente, mes): Train ({train_dup}), Test ({test_dup})")

    # 5. Validación Temporal (Meses)
    print("\n--- 4. Validación de Ventanas Temporales ('mes') ---")
    train_months = sorted(train['mes'].unique())
    test_months = sorted(test['mes'].unique())
    print(f"   - Meses en Train ({len(train_months)}): {train_months}")
    print(f"   - Meses en Test ({len(test_months)}):  {test_months}")
    
    if max(train_months) < min(test_months):
        print("✅ El split temporal es estricto: Train antecede cronológicamente a Test.")
    else:
        print("⚠️ Advertencia: Hay solapamiento temporal entre Train y Test.")

    # 6. Clientes y Solapamiento
    print("\n--- 5. Análisis de Clientes (id_cliente) ---")
    unique_clients_train = train['id_cliente'].nunique()
    unique_clients_test = test['id_cliente'].nunique()
    overlap_clients = len(set(train['id_cliente']).intersection(set(test['id_cliente'])))
    new_in_test = unique_clients_test - overlap_clients

    print(f"   - Clientes únicos en Train: {unique_clients_train:,}")
    print(f"   - Clientes únicos en Test:  {unique_clients_test:,}")
    print(f"   - Clientes de Test que aparecen en Train: {overlap_clients:,} ({overlap_clients/unique_clients_test*100:.1f}%)")
    print(f"   - Clientes nuevos en Test (Cold Start):  {new_in_test:,} ({new_in_test/unique_clients_test*100:.1f}%)")

    # 7. Variable Objetivo (Target)
    print("\n--- 6. Variable Objetivo ('objetivo') ---")
    target_counts = train['objetivo'].value_counts()
    target_prop = train['objetivo'].value_counts(normalize=True)
    print(f"   - Distribución de Clases:")
    for val, count in target_counts.items():
        print(f"     Clase {val}: {count:,} ({target_prop[val]*100:.2f}%)")
    
    # 8. Consistencia de Variables Categóricas
    print("\n--- 7. Consistencia de Categorías (Train vs Test) ---")
    cat_cols = train.select_dtypes(include=['object', 'bool']).columns.tolist()
    if 'objetivo' in cat_cols:
        cat_cols.remove('objetivo')

    unseen_found = False
    for col in cat_cols:
        train_cats = set(train[col].dropna().unique())
        test_cats = set(test[col].dropna().unique())
        unseen = test_cats - train_cats
        if unseen:
            unseen_found = True
            print(f"   ⚠️ Columna '{col}' tiene categorías en Test no vistas en Train: {unseen}")
    if not unseen_found:
        print("✅ Todas las categorías en Test existen previamente en Train.")

    # 9. Verificación de Sample Submission
    print("\n--- 8. Verificación de Sample Submission ---")
    sub_cols = list(sub.columns)
    if sub_cols == ['id_cliente', 'prediccion']:
        print("✅ Columnas de submission correctas: ['id_cliente', 'prediccion'].")
    else:
        print(f"⚠️ Columnas de submission incorrectas: {sub_cols}")

    if (sub['id_cliente'].values == test['id_cliente'].values).all():
        print("✅ El orden y los valores de 'id_cliente' en submission coinciden exactamente con test.csv.")
    else:
        print("⚠️ El orden o los IDs de 'id_cliente' no coinciden entre submission y test.")

    print("\n" + "=" * 60)
    print("✨ DIAGNÓSTICO FINAL: Dataset en excelente estado y listo para modelar.")
    print("=" * 60)

if __name__ == '__main__':
    check_dataset()
