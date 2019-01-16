import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json

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
    in_scaler = preprocessing.MinMaxScaler()
    #in_scaler = preprocessing.StandardScaler()
    input_np = in_scaler.fit_transform(input_df)

    label_df=df[labels]
    out_scaler = preprocessing.MinMaxScaler()
    #out_scaler = preprocessing.StandardScaler()
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

def read_hdf_data(path = 'premix_data',key='of_tables',in_labels=['zeta','f','pv'], labels = ['T'],scaler=None):
    # read in the hdf5 file
    try:
        df = pd.read_hdf(path,key=key) 
    except:
        print('Check the data path and key') 

    input_df=df[in_labels]

    if scaler=='MinMax':
        in_scaler = preprocessing.MinMaxScaler()
        out_scaler = preprocessing.MinMaxScaler()
    elif scaler=='Standard':
        in_scaler = preprocessing.StandardScaler()
        out_scaler = preprocessing.StandardScaler()
    else:
        raise ValueError('Only possible scalers are: MinMax or Standard.')

    input_np = in_scaler.fit_transform(input_df)

    label_df=df[labels]

    label_np = out_scaler.fit_transform(label_df)
    print('\n*******************************')
    print('The scaler is %s\n' % scaler)
    print('This is the order of the labels:')
    [print(f) for f in labels]
    print('*******************************\n')
    return input_np, label_np, df, in_scaler, out_scaler


def read_hdf_data_psi(path = 'premix_data', key='of_tables', in_labels=['zeta','f','pv'], labels = ['T'], scaler = None):
    # read in the hdf5 file
    # AND COMPUTE PSI OF THE MIXTURE
    try:
        df = pd.read_hdf(path,key=key)
    except:
        print('Check the data path and key')

    # read the molar weigths
    with open('molar_weights.json', 'r') as fp:
        molar_weights = json.load(fp)

    # read in the order of the species names
    with open('GRI_species_order') as f:
         all_species = f.read().splitlines()

    # numpy array of species molar weights
    molar_weights_np = np.array([molar_weights[s] for s in all_species])
    molar_weights_np = molar_weights_np/ 1000   # conversion from g to kg! This is needed for OpenFOAM
    T_vector = df['T'].as_matrix()

    # convert to ndarray
    gri_mass_frac = df[all_species].as_matrix()

    # COMPUTE THE CORRECT PSI VALUE
    R_universal = 8.314459
    psi_list = []

    print('Starting to compute psi ... ')
    # iterate over all rows
    for index in range(0,df.shape[0]):
        R_m = R_universal * sum(gri_mass_frac[index,:] / molar_weights_np)
        #df['psi'].iloc[index] = 1 / (R_m * row['T'])
        psi_list.append(1/(R_m * T_vector[index]))
        # print(index)

    # hand back the data to df
    df['psi'] = psi_list
    print('Done with psi!\n')

    input_df=df[in_labels]

    if scaler=='MinMax':
        in_scaler = preprocessing.MinMaxScaler()
        out_scaler = preprocessing.MinMaxScaler()
    elif scaler=='Standard':
        in_scaler = preprocessing.StandardScaler()
        out_scaler = preprocessing.StandardScaler()
    else:
        raise ValueError('Only possible scalers are: MinMax or Standard.')

    input_np = in_scaler.fit_transform(input_df)

    label_df=df[labels]

    label_np = out_scaler.fit_transform(label_df)
    print('\n*******************************')
    print('The scaler is %s\n' % scaler)
    print('This is the order of the labels:')
    [print(f) for f in labels]
    print('*******************************\n')
    return input_np, label_np, df, in_scaler, out_scaler


