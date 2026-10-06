import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_pipeline():
    print("=" * 80)
    print("🚀 PIPELINE DATA FEST - BCP: AGRUPACIONES CATEGÓRICAS Y OPTUNA LEAF-WISE")
    print("=" * 80)

    # =========================================================================
    # FASE 1: PREPROCESAMIENTO ESTRICTO Y TRATAMIENTO DE OUTLIERS
    # =========================================================================
    print("\n--- [FASE 1] Carga de Datos y Tratamiento de Outliers ---")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')

    test_ids = test_df['id_cliente'].copy()
    y = train_df['objetivo'].copy()

    cols_to_drop = ['id_cliente', 'mes']
    X_train = train_df.drop(columns=cols_to_drop + ['objetivo']).copy()
    X_test = test_df.drop(columns=cols_to_drop).copy()

    # Tratamiento de Outliers
    outlier_cols = ['ingresos', 'saldo_promedio', 'ratio_deuda_ingresos', 'distancia_sucursal_km']
    for col in outlier_cols:
        upper_limit = X_train[col].quantile(0.995)
        lower_limit = X_train[col].quantile(0.001)
        X_train[col] = X_train[col].clip(lower=lower_limit, upper=upper_limit)
        X_test[col] = X_test[col].clip(lower=lower_limit, upper=upper_limit)
        
        X_train[f'log_{col}'] = np.log1p(np.maximum(0, X_train[col]))
        X_test[f'log_{col}'] = np.log1p(np.maximum(0, X_test[col]))

    print(f"✅ Outliers tratados y variables log1p añadidas.")

    # =========================================================================
    # FASE 2: FEATURE ENGINEERING CON AGRUPACIONES CATEGÓRICAS
    # =========================================================================
    print("\n--- [FASE 2] Feature Engineering con Estadísticas y Comparativas Grupales ---")
    
    def generate_features(train, test):
        tr = train.copy()
        te = test.copy()
        
        # 1. Ratios y Métricas Financieras Básicas
        for d in [tr, te]:
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

        # 2. Cruces de Categorías
        for d in [tr, te]:
            d['ocupacion_region'] = d['ocupacion'].astype(str) + '_' + d['region'].astype(str)
            d['canal_banda'] = d['canal_adquisicion'].astype(str) + '_' + d['banda_riesgo'].astype(str)
            d['ocupacion_riesgo'] = d['ocupacion'].astype(str) + '_' + d['banda_riesgo'].astype(str)

        # 3. Agrupaciones por 'region' (Promedio Regional y Comparativa Individual)
        grp_region = tr.groupby('region')[['saldo_promedio', 'ingresos', 'ratio_deuda_ingresos']].agg(['mean', 'median', 'std']).reset_index()
        grp_region.columns = ['region'] + [f'grp_region_{col}_{stat}' for col in ['saldo_promedio', 'ingresos', 'ratio_deuda_ingresos'] for stat in ['mean', 'median', 'std']]
        tr = tr.merge(grp_region, on='region', how='left')
        te = te.merge(grp_region, on='region', how='left')

        for d in [tr, te]:
            # Diferencia y flag respecto al promedio de la región
            d['saldo_diff_region'] = d['saldo_promedio'] - d['grp_region_saldo_promedio_mean']
            d['saldo_ratio_region'] = d['saldo_promedio'] / (d['grp_region_saldo_promedio_mean'] + 1.0)
            d['saldo_por_encima_region'] = (d['saldo_promedio'] > d['grp_region_saldo_promedio_mean']).astype(int)
            
            d['ingresos_diff_region'] = d['ingresos'] - d['grp_region_ingresos_mean']
            d['ingresos_ratio_region'] = d['ingresos'] / (d['grp_region_ingresos_mean'] + 1.0)
            d['ingresos_por_encima_region'] = (d['ingresos'] > d['grp_region_ingresos_mean']).astype(int)

        # 4. Agrupaciones por 'ocupacion' (Promedio por Ocupación y Comparativa)
        grp_ocup = tr.groupby('ocupacion')[['saldo_promedio', 'ingresos', 'ratio_deuda_ingresos']].agg(['mean', 'median']).reset_index()
        grp_ocup.columns = ['ocupacion'] + [f'grp_ocup_{col}_{stat}' for col in ['saldo_promedio', 'ingresos', 'ratio_deuda_ingresos'] for stat in ['mean', 'median']]
        tr = tr.merge(grp_ocup, on='ocupacion', how='left')
        te = te.merge(grp_ocup, on='ocupacion', how='left')

        for d in [tr, te]:
            d['saldo_diff_ocup'] = d['saldo_promedio'] - d['grp_ocup_saldo_promedio_mean']
            d['saldo_ratio_ocup'] = d['saldo_promedio'] / (d['grp_ocup_saldo_promedio_mean'] + 1.0)
            d['saldo_por_encima_ocup'] = (d['saldo_promedio'] > d['grp_ocup_saldo_promedio_mean']).astype(int)

            d['ingresos_diff_ocup'] = d['ingresos'] - d['grp_ocup_ingresos_mean']
            d['ingresos_ratio_ocup'] = d['ingresos'] / (d['grp_ocup_ingresos_mean'] + 1.0)
            d['ingresos_por_encima_ocup'] = (d['ingresos'] > d['grp_ocup_ingresos_mean']).astype(int)

        # 5. Agrupaciones por '(ocupacion, region)'
        grp_ocup_reg = tr.groupby(['ocupacion', 'region'])[['saldo_promedio', 'ingresos']].agg(['mean', 'std']).reset_index()
        grp_ocup_reg.columns = ['ocupacion', 'region'] + [f'grp_ocup_reg_{col}_{stat}' for col in ['saldo_promedio', 'ingresos'] for stat in ['mean', 'std']]
        tr = tr.merge(grp_ocup_reg, on=['ocupacion', 'region'], how='left')
        te = te.merge(grp_ocup_reg, on=['ocupacion', 'region'], how='left')

        for d in [tr, te]:
            d['ratio_saldo_vs_ocup_reg'] = d['saldo_promedio'] / (d['grp_ocup_reg_saldo_promedio_mean'] + 1.0)
            d['ratio_ingresos_vs_ocup_reg'] = d['ingresos'] / (d['grp_ocup_reg_ingresos_mean'] + 1.0)

        # 6. Agrupaciones por 'canal_adquisicion' y 'banda_riesgo'
        grp_canal = tr.groupby('canal_adquisicion')[['saldo_promedio', 'ingresos']].agg(['mean']).reset_index()
        grp_canal.columns = ['canal_adquisicion', 'grp_canal_saldo_mean', 'grp_canal_ingresos_mean']
        tr = tr.merge(grp_canal, on='canal_adquisicion', how='left')
        te = te.merge(grp_canal, on='canal_adquisicion', how='left')

        for d in [tr, te]:
            d['ratio_saldo_vs_canal'] = d['saldo_promedio'] / (d['grp_canal_saldo_mean'] + 1.0)

        return tr, te

    X_train, X_test = generate_features(X_train, X_test)

    cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal',
                'ocupacion_region', 'canal_banda', 'ocupacion_riesgo']

    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    print(f"✅ Total de predictores tras feature engineering: {X_train.shape[1]}")

    # =========================================================================
    # FASE 3: OPTUNA LEAF-WISE (max_depth = -1, num_leaves entre 31 y 255)
    # =========================================================================
    print("\n--- [FASE 3] Búsqueda Optuna Leaf-Wise (max_depth = -1) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'max_depth': -1,  # Crecimiento libre leaf-wise guiado por num_leaves
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 15, 120),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
            'n_estimators': 2000,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        auc_scores = []
        for train_idx, val_idx in skf.split(X_train, y):
            X_tr, y_tr = X_train.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            preds = model.predict_proba(X_va)[:, 1]
            auc_scores.append(roc_auc_score(y_va, preds))
            
        return np.mean(auc_scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    print("⏳ Optimizando en 80 ensayos con crecimiento leaf-wise...")
    study.optimize(objective, n_trials=80, show_progress_bar=True)

    best_lgb_params = study.best_params
    print(f"\n✅ Mejores Parámetros LightGBM (AUC CV: {study.best_value:.5f}):")
    for k, v in best_lgb_params.items():
        print(f"   - {k}: {v}")

    # =========================================================================
    # FASE 4: ENTRENAMIENTO FINAL Y ENSAMBLAJE (LIGHTGBM + CATBOOST EN 5 FOLDS)
    # =========================================================================
    print("\n--- [FASE 4] Entrenamiento Final de Modelos en 5 Folds ---")

    # 1. LightGBM Final
    print("1️⃣ Entrenando LightGBM...")
    oof_lgb = np.zeros(len(X_train))
    test_preds_lgb = np.zeros(len(X_test))

    final_lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'n_estimators': 3000,
        'subsample_freq': 1,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
        **best_lgb_params
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y.iloc[val_idx]

        model_lgb = lgb.LGBMClassifier(**final_lgb_params)
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
        test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / skf.n_splits
        print(f"   Fold {fold + 1} LightGBM AUC: {roc_auc_score(y_va, oof_lgb[val_idx]):.5f}")

    print(f"   👉 AUC OOF Total LightGBM: {roc_auc_score(y, oof_lgb):.5f}")

    # 2. CatBoost Final
    print("\n2️⃣ Entrenando CatBoost con variables categóricas nativas...")
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    for col in cat_cols:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    oof_cb = np.zeros(len(X_train))
    test_preds_cb = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_cb, y)):
        X_tr, y_tr = X_train_cb.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_train_cb.iloc[val_idx], y.iloc[val_idx]

        model_cb = CatBoostClassifier(
            eval_metric='AUC',
            random_seed=42,
            early_stopping_rounds=50,
            verbose=False,
            thread_count=-1
        )

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_va, y_va, cat_features=cat_cols)
        test_pool = Pool(X_test_cb, cat_features=cat_cols)

        model_cb.fit(train_pool, eval_set=val_pool, use_best_model=True)

        oof_cb[val_idx] = model_cb.predict_proba(val_pool)[:, 1]
        test_preds_cb += model_cb.predict_proba(test_pool)[:, 1] / skf.n_splits
        print(f"   Fold {fold + 1} CatBoost AUC: {roc_auc_score(y_va, oof_cb[val_idx]):.5f}")

    print(f"   👉 AUC OOF Total CatBoost: {roc_auc_score(y, oof_cb):.5f}")

    # =========================================================================
    # FASE 5: GENERACIÓN DE LA ENTREGA (SUBMISSION)
    # =========================================================================
    print("\n--- [FASE 5] Ensamblaje y Generación de Submission ---")
    oof_ensemble = (oof_lgb + oof_cb) / 2.0
    final_test_predictions = (test_preds_lgb + test_preds_cb) / 2.0

    final_auc = roc_auc_score(y, oof_ensemble)
    final_gini = 2.0 * final_auc - 1.0

    print("=" * 60)
    print(f"📊 MÉTRICAS FINALES ESTIMADAS (Validación Cruzada OOF):")
    print(f"   ⭐ AUC Estimado:  {final_auc:.5f}")
    print(f"   🏆 Gini Estimado: {final_gini:.5f}")
    print("=" * 60)

    submission_df = pd.DataFrame({
        'id_cliente': test_ids,
        'prediccion': final_test_predictions
    })

    sub_path = 'submission_final.csv'
    submission_df.to_csv(sub_path, index=False)
    print(f"\n✅ Archivo de entrega guardado: '{sub_path}'")
    print(submission_df.head())

if __name__ == '__main__':
    run_pipeline()
