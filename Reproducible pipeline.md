
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
jupytext --set-formats ipynb,py Preprocessing.ipynb
```

## Turn py file into notebook

```
jupytext --to notebook notebook.py              # convert notebook.py to an .ipynb file with no outputs
```