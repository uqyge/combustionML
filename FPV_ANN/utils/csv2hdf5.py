import os
import pandas as pd

root_dir = './data/Flamelet_800grid'
df_fpv=pd.DataFrame()
for file in os.listdir(root_dir):
    print(file)
    df = pd.read_csv(os.path.join(root_dir,file))
    df.to_hdf('flamelet800_data.H5',key=file.split('.csv')[0],complib='zlib',complevel=9)
    # df['f']=float(file.split('.csv')[0].split('_')[1])
    df_fpv=df_fpv.append(df)

df_fpv.to_hdf('flamelet800_df.H5',key='df',complib='zlib',complevel=9)