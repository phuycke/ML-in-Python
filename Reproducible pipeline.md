## Reproducible pipeline with conda

To install have the same packages between different computers:

Configure conda distribution to use `conda-forge` channel:
- on Linux or Mac:
```
export PATH="/home/user/miniconda3/bin:$PATH"
```



```
conda --version
conda update conda
conda config --add channels conda-forge
conda config --set channel\_priority strict
```

Create virtual environment from yaml file
```
conda env create -n "pyml" python=3.8.10 -f environment.yml
```

Delete this environment
```
conda remove --name pyml --all
```

Use conda kernels systemwide:
- Install to base
```
conda install nb_conda_kernels
```

## reproducible pipeline with pyenv
- install pyenv
- install pyenv-virtualenv plugin 
  - https://github.com/pyenv/pyenv-virtualenv
-  Use pyenv to manage environment
```
pyenv install  3.8.10
pyenv virtualenv 3.8.10 ml-in-python
pyenv local ml-in-python
```
More about pyenv here https://realpython.com/intro-to-pyenv/

Activate this environment

```
pyenv activate ml-in-python
```


Install or upgrade all packages using pip 
```
pip install -r requirements.txt --upgrade
```

-  update the list  of the packages used inside virtualenv
```
pip freeze -l > requirements.txt 
```

Install kernel & add it

```
python3 -m ipykernel install --user --name=ml-in-python
```



### Optional: Zsh Setup
Put this into `.zshrc`

```
export PYENV_VIRTUALENV_DISABLE_PROMPT=1
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

if which pyenv > /dev/null; then
    eval "$(pyenv init --path)" # this only sets up the path stuff
    eval "$(pyenv init -)"      # this makes pyenv work in the shell
    alias pyenv='nocorrect pyenv'
fi
if which pyenv-virtualenv-init > /dev/null; then
    eval "$(pyenv virtualenv-init - zsh)"
fi
```

### Working with jupytext to create  synced py files fro jupyter notebook

Activate this environment

```
pyenv activate ml-in-python
```

# Turn notebook.ipynb into a paired ipynb/py notebook
```
jupytext --set-formats ipynb,py LoadData.ipynb
```