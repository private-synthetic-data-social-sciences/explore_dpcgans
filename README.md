# Exploring DP-CGANS

The purpose of this repo is to see how to use the generator in DP_CGANS and adapt it for usage with `TAPAS`.

### Setup

The following works on the snellius supercomputer.

Load the modules and make the environment. This needs to be done only once.

```bash
module load 2021
module load Python/3.9.5-GCCcore-10.3.0-bare # the non-bare version comes with an older poetry version
curl -sSL https://install.python-poetry.org | python3 - --git https://github.com/python-poetry/poetry.git@master # as of 17/10/23. https://github.com/python-poetry/poetry/issues/1917#issuecomment-1745782011

cd projects/gans/explore_dpcgans
poetry config virtualenvs.in-project true
poetry install --no-root

# sometimes the two things below are also necessary
# poetry lock 
# poetry install 
```

### Exploring the poetry installation in the login node
To see installed versions etc, one can work on the login node. But in order to activate the poetry environment, one still needs to do:
```bash
module load 2021
module load Python/3.9.5-GCCcore-10.3.0-bare

```

I think this is because poetry cannot work without a python version.

### Run an interactive GPU job 

```bash
srun -p gpu --nodes=1 --gpus=1 --time=15:00 --pty /bin/bash

cd projects/gans/explore_dpcgans
module load 2021
module load Python/3.9.5-GCCcore-10.3.0-bare

poetry shell 

# run some script

```

### To do
- also try: using as many packages as possible from pyproject directly from the module environment? how does poetry resolve such requirements?
- figure out how to copy over the file/make files accessible on the node


## Profiling the performance of DP CGAN

Start with
```bash
poetry shell
```

Then
```bash
python -m cProfile -o profiling_X.txt train_sample_dpcgans.py 
```
where `profiling_X.txt` refers to a training run with specification `X` -- at the moment, this is just `10kobs`, ie, 10_000 observations. (see the script for details).

Then, run

```python
python -i inspect_profiling_results.py
```
