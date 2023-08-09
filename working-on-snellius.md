

load the modules

```bash
module load 2021 
module load Python/3.9.5-GCCcore-10.3.0

pip install poetry # (this could also be earlier?)

poetry install 
# sometimes the two things below are also necessary
# poetry lock 
# poetry install 
```
By default, the env is installed in `~/.cache/pypoetry/virtualenvs/dp-cgans-0UW1Cbix-py3.9`

```bash
srun -p gpu --nodes=1 --gpus=1 --time=15:00 --pty /bin/bash

module load 2021
module load Python/3.9.5-GCCcore-10.3.0

poetry shell 

module load cuda11.2/toolkit 
pip uninstall nvidia_cublas_cu11 # https://stackoverflow.com/questions/74394695/how-does-one-fix-when-torch-cant-find-cuda-error-version-libcublaslt-so-11-no

```

also try: using as many packages as possible from pyproject directly from the module environment? how does poetry resolve such requirements?

