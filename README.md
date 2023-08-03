# Exploring DP-CGANS

The purpose of this repo is to see how to use the generator in DP_CGANS and adapt it for usage with `TAPAS`.

### How to work with this repo on the DAS-6 cluster

### 1. General setup

This is necessary only once. Install pyenv and the right python version. 

```bash
# venv setup 
curl https://pyenv.run | bash
```

Then add this to ~/.bashrc
```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

And install the required python version

```bash
pyenv install 3.9
pip install poetry # (this could also be earlier?)
```

Along the way I was getting the following error messages, but have ignored it for now and not encountered other problems:

```bash
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/fhafner/.pyenv/versions/3.9.17/lib/python3.9/curses/__init__.py", line 13, in <module>
    from _curses import *
ModuleNotFoundError: No module named '_curses'
WARNING: The Python curses extension was not compiled. Missing the ncurses lib?
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'readline'
WARNING: The Python readline extension was not compiled. Missing the GNU readline lib?
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/fhafner/.pyenv/versions/3.9.17/lib/python3.9/sqlite3/__init__.py", line 57, in <module>
    from sqlite3.dbapi2 import *
  File "/home/fhafner/.pyenv/versions/3.9.17/lib/python3.9/sqlite3/dbapi2.py", line 27, in <module>
    from _sqlite3 import *
ModuleNotFoundError: No module named '_sqlite3'
WARNING: The Python sqlite3 extension was not compiled. Missing the SQLite3 lib?
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/fhafner/.pyenv/versions/3.9.17/lib/python3.9/tkinter/__init__.py", line 37, in <module>
    import _tkinter # If this fails your Python may not be configured for Tk
ModuleNotFoundError: No module named '_tkinter'
WARNING: The Python tkinter extension was not compiled and GUI subsystem has been detected. Missing the Tk toolkit?
```

### 2. Create venv

**todo: forgot to create the venv in the local directory. <mark>fix this.**</mark>

```bash
# git clone https://github.com/sunchang0124/dp_cgans.git
cd explore_dpcgans

pyenv local 3.9
poetry install 
# sometimes the two things below are also necessary
# poetry lock 
# poetry install 
```

#### 3. Start a compute node and run python
```bash
# the following failed in one or the other way
# srun -N 1 -n 1 -C TitanX --gres=gpu:1 --pty bash -i # `-C TitanX`  seems invalid
# srun -N 1 -n 1  --gpus=1 --pty bash -i # this works but cuda is not available 
# srun -N 1 -n 1 --gres=gpu:1 --time=15 --pty bash -i
srun -N 1 --gres=gpu:1 --time=15 --pty bash -i

module load cuda11.2/toolkit 
pip uninstall nvidia_cublas_cu11 # https://stackoverflow.com/questions/74394695/how-does-one-fix-when-torch-cant-find-cuda-error-version-libcublaslt-so-11-no


```

Then run the python script.
