import os
import numpy as np
import pandas as pd

import json
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler


class data_scaler(object):
    def __init__(self):
        self.norm = None
        self.norm_1 = None
        self.std = None
        self.case = None
        self.scale = 1
        self.bias = 1e-20
#         self.bias = 1


        self.switcher = {
            'min_std': 'min_std',
            'std2': 'std2',
            'std_min':'std_min',
            'min': 'min',
            'no':'no',
            'log': 'log',
            'log_min':'log_min',
            'log_std':'log_std',
            'log2': 'log2',
            'sqrt_std': 'sqrt_std',
            'cbrt_std': 'cbrt_std',
            'nrt_std':'nrt_std',
            'tan': 'tan'
        }

    def fit_transform(self, input_data, case):
        self.case = case
        if self.switcher.get(self.case) == 'min_std':
            self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = self.norm.fit_transform(input_data)
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == 'std2':
            self.std = StandardScaler()
            out = self.std.fit_transform(input_data)

        if self.switcher.get(self.case) == 'std_min':
            self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = self.std.fit_transform(input_data)
            out = self.norm.fit_transform(out)

        if self.switcher.get(self.case) == 'min':
            self.norm = MinMaxScaler()
            out = self.norm.fit_transform(input_data)

        if self.switcher.get(self.case) == 'no':
            self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = input_data

        if self.switcher.get(self.case) == 'log_min':
            out = - np.log(np.asarray(input_data / self.scale) + self.bias)
            self.norm = MinMaxScaler()
            out = self.norm.fit_transform(out)

        if self.switcher.get(self.case) == 'log_std':
            out = - np.log(np.asarray(input_data / self.scale) + self.bias)
            self.std = StandardScaler()
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == 'log2':
            self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = self.norm.fit_transform(input_data)
            out = np.log(np.asarray(out) + self.bias)
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == 'sqrt_std':
            out = np.sqrt(np.asarray(input_data / self.scale))
            self.std = StandardScaler()
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == 'cbrt_std':
            out = np.cbrt(np.asarray(input_data / self.scale))
            self.std = StandardScaler()
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == 'nrt_std':
            out = np.power(np.asarray(input_data / self.scale),1/4)
            self.std = StandardScaler()
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == 'tan':
            self.norm = MaxAbsScaler()
            self.std = StandardScaler()
            out = self.std.fit_transform(input_data)
            out = self.norm.fit_transform(out)
            out = np.tan(out / (2 * np.pi + self.bias))

        return out

    def transform(self, input_data):
        if self.switcher.get(self.case) == 'min_std':
            out = self.norm.transform(input_data)
            out = self.std.transform(out)

        if self.switcher.get(self.case) == 'std2':
            out = self.std.transform(input_data)

        if self.switcher.get(self.case) == 'std_min':
            out = self.std.transform(input_data)
            out = self.norm.transform(out)

        if self.switcher.get(self.case) == 'min':
            out = self.norm.transform(input_data)

        if self.switcher.get(self.case) == 'no':
            out = input_data

        if self.switcher.get(self.case) == 'log_min':
            out = - np.log(np.asarray(input_data / self.scale) + self.bias)
            out = self.norm.transform(out)

        if self.switcher.get(self.case) == 'log_std':
            out = - np.log(np.asarray(input_data / self.scale) + self.bias)
            out = self.std.transform(out)

        if self.switcher.get(self.case) == 'log2':
            out = self.norm.transform(input_data)
            out = np.log(np.asarray(out) + self.bias)
            out = self.std.transform(out)

        if self.switcher.get(self.case) == 'sqrt_std':
            out = np.sqrt(np.asarray(input_data / self.scale))
            out = self.std.transform(out)

        if self.switcher.get(self.case) == 'cbrt_std':
            out = np.cbrt(np.asarray(input_data / self.scale))
            out = self.std.transform(out)

        if self.switcher.get(self.case) == 'nrt_std':
            out = np.power(np.asarray(input_data / self.scale),1/4)
            out = self.std.transform(out)

        if self.switcher.get(self.case) == 'tan':
            out = self.std.transform(input_data)
            out = self.norm.transform(out)
            out = np.tan(out / (2 * np.pi + self.bias))

        return out

    def inverse_transform(self, input_data):

        if self.switcher.get(self.case) == 'min_std':
            out = self.std.inverse_transform(input_data)
            out = self.norm.inverse_transform(out)

        if self.switcher.get(self.case) == 'std2':
            out = self.std.inverse_transform(input_data)

        if self.switcher.get(self.case) == 'std_min':
            out = self.norm.inverse_transform(input_data)
            out = self.std.inverse_transform(out)

        if self.switcher.get(self.case) == 'min':
            out = self.norm.inverse_transform(input_data)

        if self.switcher.get(self.case) == 'no':
            out = input_data

        if self.switcher.get(self.case) == 'log_min':
            out = self.norm.inverse_transform(input_data)
            out = (np.exp(-out) - self.bias) * self.scale

        if self.switcher.get(self.case) == 'log_std':
            out = self.std.inverse_transform(input_data)
            out = (np.exp(-out) - self.bias) * self.scale

        if self.switcher.get(self.case) == 'log2':
            out = self.std.inverse_transform(input_data)
            out = np.exp(out) - self.bias
            out = self.norm.inverse_transform(out)

        if self.switcher.get(self.case) == 'sqrt_std':
            out = self.std.inverse_transform(input_data)
            out = np.power(out,2) * self.scale

        if self.switcher.get(self.case) == 'cbrt_std':
            out = self.std.inverse_transform(input_data)
            out = np.power(out,3) * self.scale

        if self.switcher.get(self.case) == 'nrt_std':
            out = self.std.inverse_transform(input_data)
            out = np.power(out,4) * self.scale

        if self.switcher.get(self.case) == 'tan':
            out = (2 * np.pi + self.bias) * np.arctan(input_data)
            out = self.norm.inverse_transform(out)
            out = self.std.inverse_transform(out)

        return out

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

def read_h5_data(fileName, input_features=['zeta','f','pv'], labels = ['T','CH4','O2','CO2','CO','H2O','H2','OH','PVs'],i_scaler='no',o_scaler='no'):
    df = pd.read_hdf(fileName)
    df = df.clip(lower=0)
    df_o = df

    df = df[(df.f <0.42)]
    # df=df[(df.f<0.43)|(df.f>0.58)]

    # df['PVs']=df['PVs']+1
    # for i in range(5):
    #     pv_101=df[df['pv']==1]
    #     pv_101['pv']=pv_101['pv']+0.002*(i+1)
    #     df = pd.concat([df,pv_101])

    print('Outputs: ',labels)

    input_df=df[input_features]
    in_scaler = data_scaler()
    input_np = in_scaler.fit_transform(input_df,i_scaler)

    label_df=df[labels]
    out_scaler = data_scaler()
    label_np = out_scaler.fit_transform(label_df,o_scaler)

    return input_np, label_np, df_o, in_scaler, out_scaler

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
    molar_weights_np = molar_weights_np / 1000
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


