# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: ml-in-python
#     language: python
#     name: ml-in-python
# ---

# %% [markdown]
# # Prepare the data set
# - briefly survey the data
# - deal with data issues:
#   - appropriatelyhandle categorical data
#   - treat missing data
#   - identify outliers, and choose whether or not to make your analysis more robust by removing these
#   

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path
import pandas as pd

def get_project_root() -> Path:
    try:
        # This will give you a Path object pointing to the parent of the folder where the script is located.
        # If your script is `/root/scripts/script.py`, the result will be an absolute path '/root'.
        return Path(__file__).parent.parent
    except NameError as error:
        # This will give you  root folder of the project if this runned through jupyter notebook
        return Path().joinpath().absolute().parents[0]
    except Exception as exception:
        print(exception)



# %%
root= get_project_root()
path_to_file = Path.joinpath(root, 'data/train_V2.csv')
print(path_to_file)
df = pd.read_csv (path_to_file)

# %%
print (df.head(10))

# %%
#from pandas.plotting import scatter_matrix
#scatter_matrix(df, figsize=(50, 50))

