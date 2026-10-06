import json

nb_path = '/Users/aron/githubRepo/modelo-datafest-bcp/test/fourthAttempt/Pipeline_PyTorch_MoE.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Buscar la primera celda de código
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Añadir las variables de entorno anti-deadlock al principio
        anti_deadlock_code = (
            "import os\n"
            "# 🛡️ PARCHE ANTI-DEADLOCK PARA MAC (APPLE SILICON)\n"
            "os.environ['OMP_NUM_THREADS'] = '1'\n"
            "os.environ['MKL_NUM_THREADS'] = '1'\n"
            "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
            "os.environ['VECLIB_MAXIMUM_THREADS'] = '1'\n"
            "os.environ['NUMEXPR_NUM_THREADS'] = '1'\n\n"
        )
        if "OMP_NUM_THREADS" not in "".join(cell['source']):
            cell['source'].insert(0, anti_deadlock_code)
        break

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Parche Anti-Deadlock aplicado.")
