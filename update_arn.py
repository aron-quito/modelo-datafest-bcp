import json

nb_path = '/Users/aron/githubRepo/modelo-datafest-bcp/test/thirdAttempt/Pipeline_ARN_Cascada.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

new_cell_source = [
    "# 5. ENTRENAMIENTO DEL SUPERVISOR NEURONAL (MLP MEJORADO)\n",
    "print(\"⚖️ Balanceando las clases para la Red Neuronal (Oversampling)...\")\n",
    "from sklearn.utils import resample\n",
    "\n",
    "# Juntamos X e Y temporalmente para balancear\n",
    "train_meta_df = pd.DataFrame(X_train_meta)\n",
    "train_meta_df['target_y'] = y_train_full.values\n",
    "\n",
    "df_majority = train_meta_df[train_meta_df['target_y'] == 0]\n",
    "df_minority = train_meta_df[train_meta_df['target_y'] == 1]\n",
    "\n",
    "# Upsample de la clase minoritaria para igualar a la mayoritaria\n",
    "df_minority_upsampled = resample(df_minority, \n",
    "                                 replace=True,     \n",
    "                                 n_samples=len(df_majority),    \n",
    "                                 random_state=42)\n",
    "\n",
    "df_upsampled = pd.concat([df_majority, df_minority_upsampled])\n",
    "\n",
    "X_train_meta_bal = df_upsampled.drop('target_y', axis=1).values\n",
    "y_train_full_bal = df_upsampled['target_y'].values\n",
    "\n",
    "print(\"🧠 Entrenando Supervisor Neuronal (MLP Classifier Profundo)...\")\n",
    "arn_supervisor = MLPClassifier(\n",
    "    hidden_layer_sizes=(256, 128, 64), # Más profundidad y neuronas\n",
    "    activation='relu',\n",
    "    solver='adam',\n",
    "    alpha=0.1,  # Mayor regularización L2 para evitar sobreajuste\n",
    "    learning_rate_init=0.0005, # Aprendizaje más lento y seguro\n",
    "    early_stopping=True,\n",
    "    validation_fraction=0.1,\n",
    "    max_iter=1000, # Aumentamos épocas máximas\n",
    "    n_iter_no_change=30, # Aumentamos la paciencia del Early Stopping\n",
    "    random_state=42,\n",
    "    verbose=True\n",
    ")\n",
    "\n",
    "arn_supervisor.fit(X_train_meta_bal, y_train_full_bal)\n",
    "print(\"\\n✅ Supervisor Neuronal entrenado con éxito.\")"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and len(cell['source']) > 0:
        if '# 5. ENTRENAMIENTO DEL SUPERVISOR NEURONAL' in cell['source'][0]:
            cell['source'] = new_cell_source
            break

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with deep architecture and balanced dataset!")
