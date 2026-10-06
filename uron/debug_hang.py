import json

nb_path = '/Users/aron/githubRepo/modelo-datafest-bcp/test/fourthAttempt/Pipeline_PyTorch_MoE.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

debug_cell = [
    "# 3. PREPARACIÓN DE TENSORES PARA PYTORCH\n",
    "import time\n",
    "print(\"⚙️ Escalando datos para PyTorch...\")\n",
    "t0 = time.time()\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),\n",
    "        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)\n",
    "    ]\n",
    ")\n",
    "print(\"ColumnTransformer instanciado.\")\n",
    "\n",
    "X_train_rn_base = preprocessor.fit_transform(X_train_full)\n",
    "print(f\"fit_transform terminado en {time.time()-t0:.2f}s\")\n",
    "X_test_rn_base = preprocessor.transform(X_test_final)\n",
    "print(f\"transform test terminado en {time.time()-t0:.2f}s\")\n",
    "\n",
    "train_tree_preds = np.column_stack([oof_lgb, oof_cb])\n",
    "test_tree_preds = np.column_stack([test_preds_lgb, test_preds_cb])\n",
    "\n",
    "X_t_feat, X_v_feat, X_t_preds, X_v_preds, y_t, y_v = train_test_split(\n",
    "    X_train_rn_base, train_tree_preds, y_train_full.values, test_size=0.15, stratify=y_train_full.values, random_state=42\n",
    ")\n",
    "print(f\"train_test_split terminado en {time.time()-t0:.2f}s\")\n",
    "\n",
    "tensor_X_t_feat = torch.tensor(X_t_feat, dtype=torch.float32)\n",
    "tensor_X_t_preds = torch.tensor(X_t_preds, dtype=torch.float32)\n",
    "tensor_y_t = torch.tensor(y_t, dtype=torch.float32).view(-1, 1)\n",
    "\n",
    "tensor_X_v_feat = torch.tensor(X_v_feat, dtype=torch.float32)\n",
    "tensor_X_v_preds = torch.tensor(X_v_preds, dtype=torch.float32)\n",
    "tensor_y_v = torch.tensor(y_v, dtype=torch.float32).view(-1, 1)\n",
    "\n",
    "tensor_X_test_feat = torch.tensor(X_test_rn_base, dtype=torch.float32)\n",
    "tensor_X_test_preds = torch.tensor(test_tree_preds, dtype=torch.float32)\n",
    "print(f\"✅ Tensores generados en {time.time()-t0:.2f}s\")"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and len(cell['source']) > 0:
        if '# 3. PREPARACIÓN DE TENSORES PARA PYTORCH' in cell['source'][0]:
            cell['source'] = debug_cell
            break

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with debug prints!")
