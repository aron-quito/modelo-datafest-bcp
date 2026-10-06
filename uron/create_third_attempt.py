import json
import os

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Tercera Iteración: Arquitectura de Supervisor Neuronal (Deep Stacking MLP)\n",
    "\n",
    "### Objetivo:\n",
    "Usar todo el poder de una **Red Neuronal Artificial (MLP)** como meta-evaluador. La red analizará todas las características originales de los clientes JUNTO con las predicciones generadas por **LightGBM** y **CatBoost**, aprendiendo a corregir los errores que los modelos de árboles no pueden resolver por sí solos.\n",
    "\n",
    "### Flujo de Datos (Prevención de Data Leakage):\n",
    "1. **Partición Inicial (Holdout):** Separamos 10% de los datos como **Test Puro**. Estos datos NADIE los toca hasta el final (ni los árboles ni la red) para sacar la Matriz de Confusión.\n",
    "2. **Entrenamiento Base (90%):** Usamos validación cruzada (K-Fold) sobre el 90% restante. Los árboles predicen sobre los folds que no vieron (OOF). Esto asegura que la predicción que le pasamos a la ARN sea un error honesto, evitando el Data Leakage.\n",
    "3. **Entrenamiento ARN:** La ARN toma las columnas originales (escaladas) + OOF de los árboles. Internamente usa el 10% de este bloque para monitorear su propio AUC por época (Early Stopping).\n",
    "4. **Evaluación Final:** Pasamos el 10% de Test Puro por los árboles, juntamos todo y la ARN da el veredicto final."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import lightgbm as lgb\n",
    "from catboost import CatBoostClassifier, Pool\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.model_selection import StratifiedKFold, train_test_split\n",
    "from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "print(\"✅ Librerías importadas.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. CARGA Y PARTICIÓN DEL DATASET\n",
    "train_df = pd.read_csv('../../dataset/train.csv')\n",
    "\n",
    "# Separamos X e Y\n",
    "X = train_df.drop(['id_cliente', 'target'], axis=1)\n",
    "y = train_df['target']\n",
    "\n",
    "# Extraemos el 10% para el Test Final (Matriz de Confusión)\n",
    "X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(\n",
    "    X, y, test_size=0.10, stratify=y, random_state=42\n",
    ")\n",
    "\n",
    "print(f\"Datos para Entrenamiento y Validación (90%): {X_train_full.shape[0]} filas\")\n",
    "print(f\"Datos para Test Puro / Matriz de Confusión (10%): {X_test_final.shape[0]} filas\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. DEFINIR VARIABLES Y PREPROCESAMIENTO\n",
    "cat_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']\n",
    "num_cols = [c for c in X.columns if c not in cat_cols]\n",
    "\n",
    "X_train_cb = X_train_full.copy()\n",
    "X_test_cb = X_test_final.copy()\n",
    "for col in cat_cols:\n",
    "    X_train_cb[col] = X_train_cb[col].astype(str)\n",
    "    X_test_cb[col] = X_test_cb[col].astype(str)\n",
    "    X_train_full[col] = X_train_full[col].astype('category')\n",
    "    X_test_final[col] = X_test_final[col].astype('category')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. FASE BASE: ENTRENAMIENTO DE ÁRBOLES EXPERTOS (OOF para prevenir Data Leakage)\n",
    "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "\n",
    "oof_lgb = np.zeros(len(X_train_full))\n",
    "test_preds_lgb = np.zeros(len(X_test_final))\n",
    "\n",
    "oof_cb = np.zeros(len(X_train_full))\n",
    "test_preds_cb = np.zeros(len(X_test_final))\n",
    "\n",
    "best_lgb_params = {'num_leaves': 42, 'learning_rate': 0.017766463699737917, 'min_child_samples': 44, 'subsample': 0.6548629928035845, 'colsample_bytree': 0.595372408030665, 'reg_alpha': 4.9893861303564696e-05, 'reg_lambda': 0.1440126277634314}\n",
    "\n",
    "print(\"🌳 Entrenando Expertos (LightGBM y CatBoost)...\")\n",
    "for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):\n",
    "    # LightGBM\n",
    "    X_tr_lgb, y_tr_lgb = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]\n",
    "    X_va_lgb, y_va_lgb = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]\n",
    "    \n",
    "    model_lgb = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=1500, random_state=42, verbose=-1, **best_lgb_params)\n",
    "    model_lgb.fit(X_tr_lgb, y_tr_lgb, eval_set=[(X_va_lgb, y_va_lgb)], callbacks=[lgb.early_stopping(30, verbose=False)])\n",
    "    oof_lgb[val_idx] = model_lgb.predict_proba(X_va_lgb)[:, 1]\n",
    "    test_preds_lgb += model_lgb.predict_proba(X_test_final)[:, 1] / skf.n_splits\n",
    "    \n",
    "    # CatBoost\n",
    "    X_tr_cb, y_tr_cb = X_train_cb.iloc[train_idx], y_train_full.iloc[train_idx]\n",
    "    X_va_cb, y_va_cb = X_train_cb.iloc[val_idx], y_train_full.iloc[val_idx]\n",
    "    \n",
    "    model_cb = CatBoostClassifier(eval_metric='AUC', random_seed=42, early_stopping_rounds=30, verbose=False, thread_count=-1)\n",
    "    train_pool = Pool(X_tr_cb, y_tr_cb, cat_features=cat_cols)\n",
    "    val_pool = Pool(X_va_cb, y_va_cb, cat_features=cat_cols)\n",
    "    test_pool = Pool(X_test_cb, cat_features=cat_cols)\n",
    "    \n",
    "    model_cb.fit(train_pool, eval_set=val_pool, use_best_model=True)\n",
    "    oof_cb[val_idx] = model_cb.predict_proba(val_pool)[:, 1]\n",
    "    test_preds_cb += model_cb.predict_proba(test_pool)[:, 1] / skf.n_splits\n",
    "\n",
    "print(f\"✅ OOF LightGBM AUC: {roc_auc_score(y_train_full, oof_lgb):.5f}\")\n",
    "print(f\"✅ OOF CatBoost AUC: {roc_auc_score(y_train_full, oof_cb):.5f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. PREPARACIÓN DEL SÚPER-DATASET PARA LA RED NEURONAL\n",
    "print(\"⚙️ Transformando columnas originales y acoplando predicciones de los árboles...\")\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),\n",
    "        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)\n",
    "    ]\n",
    ")\n",
    "\n",
    "# Ajustar escalador solo en datos de entrenamiento\n",
    "X_train_rn_base = preprocessor.fit_transform(X_train_full)\n",
    "X_test_rn_base = preprocessor.transform(X_test_final)\n",
    "\n",
    "# Concatenamos [Features Originales] + [Opinion LGBM] + [Opinion CatBoost]\n",
    "X_train_meta = np.column_stack([X_train_rn_base, oof_lgb, oof_cb])\n",
    "X_test_meta = np.column_stack([X_test_rn_base, test_preds_lgb, test_preds_cb])\n",
    "\n",
    "print(f\"Dimensiones finales del Súper-Dataset: {X_train_meta.shape[1]} columnas (features + opiniones)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. ENTRENAMIENTO DEL SUPERVISOR NEURONAL (MLP)\n",
    "print(\"🧠 Entrenando Supervisor Neuronal (MLP Classifier)...\")\n",
    "# Usamos early_stopping=True que internamente separa el 10% (validation_fraction=0.1) \n",
    "# de X_train_meta para frenar el entrenamiento si el AUC baja.\n",
    "arn_supervisor = MLPClassifier(\n",
    "    hidden_layer_sizes=(128, 64, 32),\n",
    "    activation='relu',\n",
    "    solver='adam',\n",
    "    alpha=0.01,\n",
    "    learning_rate_init=0.001,\n",
    "    early_stopping=True,\n",
    "    validation_fraction=0.1,\n",
    "    max_iter=150,\n",
    "    random_state=42,\n",
    "    verbose=True\n",
    ")\n",
    "\n",
    "arn_supervisor.fit(X_train_meta, y_train_full)\n",
    "print(\"\\n✅ Supervisor Neuronal entrenado con éxito.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6. EVALUACIÓN FINAL SOBRE EL 10% TEST PURO\n",
    "print(\"\\n=======================================================\")\n",
    "print(\"📊 RESULTADOS FINALES SOBRE EL SET DE PRUEBA (10% PURO)\")\n",
    "print(\"=======================================================\")\n",
    "\n",
    "# La ARN emite su veredicto final\n",
    "final_probs = arn_supervisor.predict_proba(X_test_meta)[:, 1]\n",
    "final_preds_bin = arn_supervisor.predict(X_test_meta)\n",
    "\n",
    "final_auc = roc_auc_score(y_test_final, final_probs)\n",
    "final_gini = 2.0 * final_auc - 1.0\n",
    "final_acc = accuracy_score(y_test_final, final_preds_bin)\n",
    "\n",
    "print(f\"⭐ AUC FINAL:  {final_auc:.5f}\")\n",
    "print(f\"🏆 GINI FINAL: {final_gini:.5f}\")\n",
    "print(f\"🎯 ACCURACY:   {final_acc:.5f}\\n\")\n",
    "\n",
    "print(\"📉 MATRIZ DE CONFUSIÓN:\")\n",
    "print(confusion_matrix(y_test_final, final_preds_bin))\n",
    "print(\"\\n📋 REPORTE DE CLASIFICACIÓN:\")\n",
    "print(classification_report(y_test_final, final_preds_bin))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (modelo-datafest-bcp)",
   "language": "python",
   "name": "modelo-datafest-bcp"
  },
  "language_info": {
   "name": "python",
   "version": "3.14.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

nb_path = '/Users/aron/githubRepo/modelo-datafest-bcp/test/thirdAttempt/Pipeline_ARN_Cascada.ipynb'
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook created successfully at {nb_path}")
