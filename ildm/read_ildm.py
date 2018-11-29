import numpy as np
import pandas as pd

df=pd.DataFrame()

T=[]

with open('./data/output.dat','r') as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]

    grids=['gridCO2','gridH2O','gridZ']
    for item in grids:
        vars()[item]=[]

    species = []
    for name in lines[1:17]:
        species.append(name.split(' ')[0])
        vars()[name.split(' ')[0]]=[]

    etc_src = ['density','sCO2','sH2O','sZ']
    for item in etc_src:
        vars()[item]=[]

    etc_nb =['ix','n1','n2','n3','n4','n5','n6','n7']
    for item in etc_nb:
        vars()[item]=[]

    ## reading data
    for line in lines[77:]:
        #grid
        if len(line.split(' ')) == 7:
            for j, src in enumerate(grids):
                vars()[src].append(int(line.split(' ')[j+1]))


        # first 10 species
        if len(line.split(' ')) == 11:
            T.append(line.split(' ')[0])
            for j in range(10):
                vars()[species[j]].append(float(line.split(' ')[j+1]))

        # following 6 species
        if len(line.split(' ')) == 6:
            for j in range(10,16):
                vars()[species[j]].append(float(line.split(' ')[j-10]))


        # source terms
        if len(line.split(' ')) == 4:
            for j, src in enumerate(etc_src):
                vars()[src].append(float(line.split(' ')[j]))

        # neighbours
        if len(line.split(' ')) == 10:
            for j, src in enumerate(etc_nb):
                vars()[src].append(int(line.split(' ')[j+1]))

        if line=='End':
            break

    for item in grids:
        vars()[item]=np.asarray(vars()[item])
        df[item]=vars()[item]

    T=np.asarray(T)
    df['T']=T
    for sp in species:
        vars()[sp]=np.asarray(vars()[sp])
        df[sp]=vars()[sp]

    for item in etc_src:
        vars()[item]=np.asarray(vars()[item])
        df[item]=vars()[item]

    for item in etc_nb:
        vars()[item]=np.asarray(vars()[item])
        df[item]=vars()[item]

    print(df.shape)
    print(df.columns)

    df.to_hdf('bigDataFrame.H5',key='ildm_raw_data',complevel=9,complib='zlib')
