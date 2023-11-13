import pandas as pd
import numpy as np
def load():
    df = pd.read_csv('/home/ibab/movedfiles/titanic.csv')
    print(df)
def rows():
    df = pd.read_csv('/home/ibab/movedfiles/titanic.csv')
    rows = len(df.axes[0])
    print("number of rows are: ", rows)
def cols():
    df = pd.read_csv('/home/ibab/movedfiles/titanic.csv')
    col = len(df.axes[1])
    print("number of columns are: ", col)
def missing():
    df = pd.read_csv('/home/ibab/movedfiles/titanic.csv')
    rep = df.replace(np.nan, 0)
    print(rep)
