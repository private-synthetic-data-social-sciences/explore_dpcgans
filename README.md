# Exploring DP-CGANS

The purpose of this repo is to see how to use the generator in DP_CGANS and adapt it for usage with `TAPAS`.

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

### Exploring the poetry installation in the login node
To see installed versions etc, one can work on the login node. But in order to activate the poetry environment, one still needs to do:
```bash
module load 2021 
module load Python/3.9.5-GCCcore-10.3.0

```

I think this is because poetry cannot work without a python version.

### Run an interactive GPU job 

```bash
srun -p gpu --nodes=1 --gpus=1 --time=15:00 --pty /bin/bash

cd projects/gans/explore_dpcgans
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

poetry shell 

python sample_dpcgans.py # raises errors

# module load cuda11.2/toolkit 
# pip uninstall nvidia_cublas_cu11 # https://stackoverflow.com/questions/74394695/how-does-one-fix-when-torch-cant-find-cuda-error-version-libcublaslt-so-11-no

```

### To do
- also try: using as many packages as possible from pyproject directly from the module environment? how does poetry resolve such requirements?
- figure out how to copy over the file/make files accessible on the node


### Problem

`poetry shell` gives the following output:
```bash
poetry shell 
Spawning shell within /gpfs/home2/hafner/projects/gans/explore_dpcgans/.venv
. /gpfs/home2/hafner/projects/gans/explore_dpcgans/.venv/bin/activate
flatpak: /sw/arch/Centos8/EB_production/2021/software/XZ/5.2.5-GCCcore-10.3.0/lib/liblzma.so.5: version `XZ_5.2' not found (required by /lib64/libarchive.so.13)
```

`python sample_dpcgans.py` gives the following output:

```bash
Traceback (most recent call last):
  File "/gpfs/home2/hafner/projects/gans/explore_dpcgans/sample_dpcgans.py", line 16, in <module>
    model.sample()
  File "/gpfs/home2/hafner/projects/gans/explore_dpcgans/.venv/lib/python3.9/site-packages/dp_cgans/base.py", line 442, in sample
    return self._sample_batch(num_rows, max_retries, max_rows_multiplier)
  File "/gpfs/home2/hafner/projects/gans/explore_dpcgans/.venv/lib/python3.9/site-packages/dp_cgans/base.py", line 299, in _sample_batch
    sampled, num_valid = self._sample_rows(
  File "/gpfs/home2/hafner/projects/gans/explore_dpcgans/.venv/lib/python3.9/site-packages/dp_cgans/base.py", line 228, in _sample_rows
    sampled = self._sample(num_rows)
  File "/gpfs/home2/hafner/projects/gans/explore_dpcgans/.venv/lib/python3.9/site-packages/dp_cgans/dp_cgan_init.py", line 80, in _sample
    return self._model.sample(num_rows)
  File "/gpfs/home2/hafner/projects/gans/explore_dpcgans/.venv/lib/python3.9/site-packages/dp_cgans/synthesizers/dp_cgan.py", line 664, in sample
    return self._transformer.inverse_transform(data)
  File "/gpfs/home2/hafner/projects/gans/explore_dpcgans/.venv/lib/python3.9/site-packages/dp_cgans/data_transformer.py", line 198, in inverse_transform
    recovered_column_data = self._inverse_transform_discrete(
  File "/gpfs/home2/hafner/projects/gans/explore_dpcgans/.venv/lib/python3.9/site-packages/dp_cgans/data_transformer.py", line 178, in _inverse_transform_discrete
    return ohe.reverse_transform(data)[column_transform_info.column_name]
  File "/gpfs/home2/hafner/projects/gans/explore_dpcgans/.venv/lib/python3.9/site-packages/rdt/transformers/base.py", line 279, in reverse_transform
    if any(column not in data.columns for column in self.output_columns):
TypeError: 'NoneType' object is not iterable
```