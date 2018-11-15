import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing


#%%
def read_csv_data(path):
    df = pd.DataFrame()
    path = 'data'
    for fn in os.listdir('./data'):
        # if os.path.isfile(fn):
        print(fn)
        tmp = pd.read_csv(os.path.join(path, fn))
        tmp['p'] = int(fn.replace('bar.csv', ''))
        df = df.append(tmp)

    input=df[['p','he']]
    # in_scaler = preprocessing.MinMaxScaler()
    in_scaler = preprocessing.StandardScaler()
    input = in_scaler.fit_transform(input)

    label=df[['rho','T','thermo:mu','Cp']]
    # out_scaler = preprocessing.MinMaxScaler()
    out_scaler = preprocessing.StandardScaler()
    label = out_scaler.fit_transform(label)
    return input, label,df, in_scaler, out_scaler

a,b,df, in_scaler, out_scaler=read_csv_data('data')

ref=df.loc[df['p']==34]
x_test = in_scaler.transform(ref[['p','he']])