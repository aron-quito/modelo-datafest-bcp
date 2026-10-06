import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb_path = '/Users/aron/githubRepo/modelo-datafest-bcp/test/secondAttempt/LightGBM.ipynb'
with open(nb_path) as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
try:
    ep.preprocess(nb, {'metadata': {'path': '/Users/aron/githubRepo/modelo-datafest-bcp/test/secondAttempt'}})
    print("Notebook ran successfully!")
except Exception as e:
    print("Error running notebook:", e)
