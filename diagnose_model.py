import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

def run_diagnostics():
    print("=" * 80)
    print("🔬 DIAGNÓSTICO PROFUNDO DE MODELOS: IMPORTANCIA, ERRORES Y CORRELACIÓN")
    print("=" * 80)

    # 1. Carga y Preprocesamiento
    print("\n[1/4] Cargando datos y aplicando Feature Engineering...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')

    y = train_df['objetivo'].copy()
    cols_to_drop = ['id_cliente', 'mes']
    X_train = train_df.drop(columns=cols_to_drop + ['objetivo']).copy()
    X_test = test_df.drop(columns=cols_to_drop).copy()

    # Outliers
    outlier_cols = ['ingresos', 'saldo_promedio', 'ratio_deuda_ingresos', 'distancia_sucursal_km']
    for col in outlier_cols:
        upper_limit = X_train[col].quantile(0.995)
        lower_limit = X_train[col].quantile(0.001)
        X_train[col] = X_train[col].clip(lower=lower_limit, upper=upper_limit)
        X_test[col] = X_test[col].clip(lower=lower_limit, upper=upper_limit)
        X_train[f'log_{col}'] = np.log1p(np.maximum(0, X_train[col]))
        X_test[f'log_{col}'] = np.log1p(np.maximum(0, X_test[col]))

    # Feature Engineering
    for d in [X_train, X_test]:
        d['capacidad_ahorro'] = d['saldo_promedio'] / (d['ingresos'] + 1.0)
        d['intensidad_productos'] = d['saldo_promedio'] / (d['numero_productos'] + 0.001)
        d['dias_inactividad_relativa'] = d['dias_ultima_transaccion'] - d['dias_ultima_interaccion']
        d['deuda_total_estimada'] = d['ingresos'] * d['ratio_deuda_ingresos']
        d['proporcion_deuda_saldo'] = d['deuda_total_estimada'] / (d['saldo_promedio'] + 1.0)
        d['saldo_neto_estimado'] = d['saldo_promedio'] - d['deuda_total_estimada']
        d['saldo_por_edad'] = d['saldo_promedio'] / (d['edad'] + 1.0)
        d['ingresos_por_edad'] = d['ingresos'] / (d['edad'] + 1.0)
        d['ratio_visitas_interaccion'] = d['visitas_web_ultimos_90_dias'] / (d['dias_ultima_interaccion'] + 1.0)
        d['total_productos_financieros'] = (
            d['numero_productos'] + 
            d['tiene_tarjeta_credito'].astype(int) + 
            d['tiene_prestamo'].astype(int) + 
            d['tiene_seguro'].astype(int)
        )
        d['score_digital'] = (
            d['activo_movil'].astype(int) * 2 + 
            (d['visitas_web_ultimos_90_dias'] > 0).astype(int) + 
            (d['dispositivo_principal'] == 'movil').astype(int)
        )
        d['ocupacion_region'] = d['ocupacion'].astype(str) + '_' + d['region'].astype(str)
        d['canal_banda'] = d['canal_adquisicion'].astype(str) + '_' + d['banda_riesgo'].astype(str)
        d['ocupacion_riesgo'] = d['ocupacion'].astype(str) + '_' + d['banda_riesgo'].astype(str)

    # Agrupaciones
    grp_region = X_train.groupby('region')[['saldo_promedio', 'ingresos', 'ratio_deuda_ingresos']].agg(['mean', 'median']).reset_index()
    grp_region.columns = ['region'] + [f'grp_region_{col}_{stat}' for col in ['saldo_promedio', 'ingresos', 'ratio_deuda_ingresos'] for stat in ['mean', 'median']]
    X_train = X_train.merge(grp_region, on='region', how='left')
    X_test = X_test.merge(grp_region, on='region', how='left')

    for d in [X_train, X_test]:
        d['saldo_diff_region'] = d['saldo_promedio'] - d['grp_region_saldo_promedio_mean']
        d['saldo_ratio_region'] = d['saldo_promedio'] / (d['grp_region_saldo_promedio_mean'] + 1.0)
        d['saldo_por_encima_region'] = (d['saldo_promedio'] > d['grp_region_saldo_promedio_mean']).astype(int)
        d['ingresos_diff_region'] = d['ingresos'] - d['grp_region_ingresos_mean']
        d['ingresos_ratio_region'] = d['ingresos'] / (d['grp_region_ingresos_mean'] + 1.0)
        d['ingresos_por_encima_region'] = (d['ingresos'] > d['grp_region_ingresos_mean']).astype(int)

    grp_ocup = X_train.groupby('ocupacion')[['saldo_promedio', 'ingresos', 'ratio_deuda_ingresos']].agg(['mean', 'median']).reset_index()
    grp_ocup.columns = ['ocupacion'] + [f'grp_ocup_{col}_{stat}' for col in ['saldo_promedio', 'ingresos', 'ratio_deuda_ingresos'] for stat in ['mean', 'median']]
    X_train = X_train.merge(grp_ocup, on='ocupacion', how='left')
    X_test = X_test.merge(grp_ocup, on='ocupacion', how='left')

    for d in [X_train, X_test]:
        d['saldo_diff_ocup'] = d['saldo_promedio'] - d['grp_ocup_saldo_promedio_mean']
        d['saldo_ratio_ocup'] = d['saldo_promedio'] / (d['grp_ocup_saldo_promedio_mean'] + 1.0)
        d['saldo_por_encima_ocup'] = (d['saldo_promedio'] > d['grp_ocup_saldo_promedio_mean']).astype(int)
        d['ingresos_diff_ocup'] = d['ingresos'] - d['grp_ocup_ingresos_mean']
        d['ingresos_ratio_ocup'] = d['ingresos'] / (d['grp_ocup_ingresos_mean'] + 1.0)
        d['ingresos_por_encima_ocup'] = (d['ingresos'] > d['grp_ocup_ingresos_mean']).astype(int)

    cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal',
                'ocupacion_region', 'canal_banda', 'ocupacion_riesgo']

    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    feature_names = list(X_train.columns)
    print(f"✅ Total predictores analizados: {len(feature_names)}")

    # 2. Entrenamiento en StratifiedKFold (5 Folds) para LightGBM y CatBoost
    print("\n[2/4] Entrenando modelos para extraer OOF y Feature Importances...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(X_train))
    oof_cb = np.zeros(len(X_train))
    feature_importances_gain = np.zeros(len(feature_names))

    # CatBoost data
    X_train_cb = X_train.copy()
    for col in cat_cols:
        X_train_cb[col] = X_train_cb[col].astype(str)

    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'num_leaves': 63,
        'learning_rate': 0.05,
        'min_child_samples': 30,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'n_estimators': 1500,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y.iloc[val_idx]

        # LightGBM
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
        feature_importances_gain += model_lgb.booster_.feature_importance(importance_type='gain') / skf.n_splits

        # CatBoost
        X_tr_cb, y_tr_cb = X_train_cb.iloc[train_idx], y.iloc[train_idx]
        X_va_cb, y_va_cb = X_train_cb.iloc[val_idx], y.iloc[val_idx]

        model_cb = CatBoostClassifier(
            eval_metric='AUC',
            random_seed=42,
            early_stopping_rounds=50,
            verbose=False,
            thread_count=-1
        )
        train_pool = Pool(X_tr_cb, y_tr_cb, cat_features=cat_cols)
        val_pool = Pool(X_va_cb, y_va_cb, cat_features=cat_cols)
        model_cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof_cb[val_idx] = model_cb.predict_proba(val_pool)[:, 1]

    auc_lgb = roc_auc_score(y, oof_lgb)
    auc_cb = roc_auc_score(y, oof_cb)
    print(f"✅ OOF LightGBM AUC: {auc_lgb:.5f} | OOF CatBoost AUC: {auc_cb:.5f}")

    # =========================================================================
    # OBJETIVO 1: FEATURE IMPORTANCE (GAIN)
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 1. FEATURE IMPORTANCE (LIGHTGBM - GANANCIA / GAIN)")
    print("=" * 80)

    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': feature_importances_gain
    }).sort_values('importance_gain', ascending=False).reset_index(drop=True)

    fi_df['importance_pct'] = (fi_df['importance_gain'] / fi_df['importance_gain'].sum()) * 100

    print("\n🔝 TOP 5 VARIABLES MÁS DOMINANTES:")
    print(fi_df.head(5).to_string(index=False))

    print("\n🔻 TOP 10 VARIABLES MÁS IGNORADAS O DE MENOR GANANCIA:")
    print(fi_df.tail(10).to_string(index=False))

    # Guardar gráfico de Feature Importance
    plt.figure(figsize=(10, 12))
    sns.barplot(data=fi_df.head(25), x='importance_gain', y='feature', palette='viridis')
    plt.title('Top 25 Feature Importances (Gain) - LightGBM', fontsize=14, fontweight='bold')
    plt.xlabel('Total Gain', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance_lightgbm.png', dpi=200)
    plt.close()
    print("📁 Gráfico guardado en 'feature_importance_lightgbm.png'.")

    # =========================================================================
    # OBJETIVO 2: ANÁLISIS DE ERRORES (FALSOS NEGATIVOS SEVEROS EN CLASE 1)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🔍 2. ANÁLISIS DE ERRORES: CLIENTES '1' INCOMPRENDIDOS (FALSOS NEGATIVOS SEVEROS)")
    print("=" * 80)

    eval_df = train_df.copy()
    eval_df['pred_lgb'] = oof_lgb
    eval_df['pred_cb'] = oof_cb
    eval_df['pred_ensemble'] = (oof_lgb + oof_cb) / 2.0
    eval_df['error_abs_lgb'] = np.abs(eval_df['objetivo'] - eval_df['pred_lgb'])

    # Filtrar solo clientes positivos reales (objetivo == 1)
    positives = eval_df[eval_df['objetivo'] == 1].copy()

    # Percentil 20 inferior de probabilidad (los peores falsos negativos) vs Percentil 20 superior (los mejor detectados)
    threshold_low = positives['pred_lgb'].quantile(0.20)
    threshold_high = positives['pred_lgb'].quantile(0.80)

    missed_positives = positives[positives['pred_lgb'] <= threshold_low]  # Incomprendidos
    caught_positives = positives[positives['pred_lgb'] >= threshold_high]  # Bien detectados

    print(f"\nTotal clientes convertidores (Clase 1): {len(positives):,}")
    print(f"❌ Incomprendidos (peores Falsos Negativos, pred <= {threshold_low:.4f}): {len(missed_positives):,} clientes")
    print(f"🎯 Bien detectados (mejores Aciertos, pred >= {threshold_high:.4f}):   {len(caught_positives):,} clientes")

    # Comparación de perfiles numéricos
    num_cols_to_compare = [
        'saldo_promedio', 'ingresos', 'ratio_deuda_ingresos', 'edad', 
        'antiguedad_cuenta_meses', 'numero_productos', 'dias_ultima_transaccion', 
        'dias_ultima_interaccion', 'visitas_web_ultimos_90_dias', 'distancia_sucursal_km'
    ]

    profile_comp = []
    for col in num_cols_to_compare:
        mean_missed = missed_positives[col].mean()
        mean_caught = caught_positives[col].mean()
        mean_all_pos = positives[col].mean()
        median_missed = missed_positives[col].median()
        median_caught = caught_positives[col].median()
        profile_comp.append({
            'Variable': col,
            'Media_Incomprendidos (FN)': round(mean_missed, 2),
            'Media_BienDetectados': round(mean_caught, 2),
            'Diferencia_Relativa (%)': round(((mean_missed - mean_caught) / (mean_caught + 1e-5)) * 100, 1),
            'Mediana_FN': round(median_missed, 2),
            'Mediana_Acierto': round(median_caught, 2)
        })

    profile_df = pd.DataFrame(profile_comp)
    print("\n📊 COMPARATIVA DE VARIABLES NUMÉRICAS:")
    print(profile_df.to_string(index=False))

    # Comparativa de variables categóricas
    print("\n📊 COMPARATIVA DE VARIABLES CATEGÓRICAS (% Distribución):")
    for cat in ['ocupacion', 'region', 'banda_riesgo', 'canal_adquisicion', 'tiene_tarjeta_credito', 'activo_movil', 'tiene_prestamo']:
        missed_dist = (missed_positives[cat].value_counts(normalize=True) * 100).round(1)
        caught_dist = (caught_positives[cat].value_counts(normalize=True) * 100).round(1)
        cat_comparison = pd.DataFrame({'% Incomprendidos (FN)': missed_dist, '% Bien Detectados': caught_dist}).fillna(0)
        print(f"\n--- {cat} ---")
        print(cat_comparison.to_string())

    # =========================================================================
    # OBJETIVO 3: MATRIZ DE CORRELACIÓN DE PREDICCIONES Y ERRORES
    # =========================================================================
    print("\n" + "=" * 80)
    print("🔗 3. MATRIZ DE CORRELACIÓN DE PREDICCIONES Y ERRORES (LIGHTGBM vs CATBOOST)")
    print("=" * 80)

    # 1. Correlación de predicciones directas
    pearson_corr, _ = pearsonr(oof_lgb, oof_cb)
    spearman_corr, _ = spearmanr(oof_lgb, oof_cb)

    # 2. Correlación de errores residuales (y - pred)
    err_lgb = y.values - oof_lgb
    err_cb = y.values - oof_cb
    err_pearson, _ = pearsonr(err_lgb, err_cb)
    err_spearman, _ = spearmanr(err_lgb, err_cb)

    print(f"📈 Correlación de Predicciones:")
    print(f"   - Pearson (lineal):    {pearson_corr * 100:.2f}% ({pearson_corr:.4f})")
    print(f"   - Spearman (rangos):   {spearman_corr * 100:.2f}% ({spearman_corr:.4f})")

    print(f"\n📉 Correlación de Errores Residuales:")
    print(f"   - Pearson de Errores:  {err_pearson * 100:.2f}% ({err_pearson:.4f})")
    print(f"   - Spearman de Errores: {err_spearman * 100:.2f}% ({err_spearman:.4f})")

    # Gráfico de dispersión y correlación de predicciones
    plt.figure(figsize=(8, 6))
    plt.hexbin(oof_lgb, oof_cb, gridsize=50, cmap='inferno', mincnt=1)
    plt.colorbar(label='Densidad de clientes')
    plt.xlabel('Predicción LightGBM OOF', fontsize=12)
    plt.ylabel('Predicción CatBoost OOF', fontsize=12)
    plt.title(f'Dispersión Predicciones LightGBM vs CatBoost (Pearson: {pearson_corr:.3f})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_lgb_catboost.png', dpi=200)
    plt.close()
    print("📁 Gráfico guardado en 'correlation_lgb_catboost.png'.")

    # Diagnóstico e Insights
    print("\n" + "=" * 80)
    print("💡 CONCLUSIONES Y DIAGNÓSTICO CLAVE:")
    print("=" * 80)

if __name__ == '__main__':
    run_diagnostics()
