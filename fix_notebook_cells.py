import json

nb_path = '/Users/aron/githubRepo/modelo-datafest-bcp/test/secondAttempt/LightGBM.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

imports_code = [
    "import os\n",
    "import warnings\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import optuna\n",
    "import lightgbm as lgb\n",
    "from catboost import CatBoostClassifier, Pool\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import roc_auc_score\n",
    "from scipy.optimize import minimize\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
    "print(\"✅ Librerías importadas correctamente.\")"
]

# Fix cell 1 (which became Phase 6)
if len(nb['cells']) > 1 and nb['cells'][1]['cell_type'] == 'code':
    nb['cells'][1]['source'] = imports_code

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Restored cell 1!")
