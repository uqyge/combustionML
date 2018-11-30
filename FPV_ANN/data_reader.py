import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing


#%%
def read_data(fileName,input_features, labels = ['T','CH4','O2','CO2','CO','H2O','H2','OH','PVs']):
    df = pd.read_hdf(fileName)

    input_df=df[input_features]
    in_scaler = preprocessing.MinMaxScaler()
    # in_scaler = preprocessing.StandardScaler()
    input_np = in_scaler.fit_transform(input_df)

    label_df=df[labels]
    out_scaler = preprocessing.MinMaxScaler()
    # out_scaler = preprocessing.StandardScaler()
    label_np = out_scaler.fit_transform(label_df)
    return input_np, label_np, df, in_scaler, out_scaler


if __name__ =="__main__":
    a,b,df, in_scaler, out_scaler=read_data('./data/fpv_df.H5')

