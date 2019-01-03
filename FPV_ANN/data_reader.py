import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing


#%%
def read_csv_data(path = 'premix_data', labels = ['T','CH4','O2','CO2','CO','H2O','H2','OH','PVs']):
    df = pd.DataFrame()
    # path = 'data'
    for fn in os.listdir(path):
        # if os.path.isfile(fn):
        print(fn)
        tmp = pd.read_csv(os.path.join(path, fn))
        tmp['f']  = float(fn[fn.find('f_') + len('f_'):fn.rfind('.csv')])
        df = df.append(tmp)

    input_df=df[['f','PV']]
    # in_scaler = preprocessing.MinMaxScaler()
    in_scaler = preprocessing.StandardScaler()
    input_np = in_scaler.fit_transform(input_df)

    label_df=df[labels]
    # out_scaler = preprocessing.MinMaxScaler()
    out_scaler = preprocessing.StandardScaler()
    label_np = out_scaler.fit_transform(label_df)
    # print('\n*******************************')
    # print('This is the order of the labels:')
    # print('rho\nT\nthermo:mu\nCp\nthermo:psi\nthermo:alpha\nthermo:as')
    # print('*******************************\n')
    return input_np, label_np, df, in_scaler, out_scaler

def read_h5_data(fileName, input_features, labels = ['T','CH4','O2','CO2','CO','H2O','H2','OH','PVs']):
    df = pd.read_hdf(fileName)

    input_df=df[input_features]
    # in_scaler = preprocessing.MinMaxScaler()
    in_scaler = preprocessing.StandardScaler()
    input_np = in_scaler.fit_transform(input_df)

    label_df=df[labels]
    # out_scaler = preprocessing.MinMaxScaler()
    out_scaler = preprocessing.StandardScaler()
    label_np = out_scaler.fit_transform(label_df)
    # print('\n*******************************')
    # print('This is the order of the labels:')
    # print('rho\nT\nthermo:mu\nCp\nthermo:psi\nthermo:alpha\nthermo:as')
    # print('*******************************\n')
    return input_np, label_np, df, in_scaler, out_scaler

if __name__ =="__main__":
    a,b,df, in_scaler, out_scaler=read_csv_data('data')

    ref=df.loc[df['p']==34]
    x_test = in_scaler.transform(ref[['p','he']])