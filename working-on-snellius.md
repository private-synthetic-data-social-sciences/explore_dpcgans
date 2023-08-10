
### Setup

The following works on the snellius supercomputer.

Load the modules and make the environment. This needs to be done only once.

```bash
module load 2021 
module load Python/3.9.5-GCCcore-10.3.0
pip install poetry 

cd projects/gans/explore_dpcgans
poetry config virtualenvs.in-project true
poetry install 

# sometimes the two things below are also necessary
# poetry lock 
# poetry install 
```


### Run a GPU job 

```bash
srun -p gpu --nodes=1 --gpus=1 --time=15:00 --pty /bin/bash

cd projects/gans/explore_dpcgans
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

poetry shell 

python sample_dpcgans.py

# module load cuda11.2/toolkit 
# pip uninstall nvidia_cublas_cu11 # https://stackoverflow.com/questions/74394695/how-does-one-fix-when-torch-cant-find-cuda-error-version-libcublaslt-so-11-no

```

### To do
- also try: using as many packages as possible from pyproject directly from the module environment? how does poetry resolve such requirements?
- figure out how to copy over the file/make files accessible on the node
