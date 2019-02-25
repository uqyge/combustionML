import pandas as pd
import numpy as np
import glob
import re
import os


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# %%
# read pv, zeta and f values
# os.remove('fg')
f_scale = 501
zeta_scale = 10
pv_scale = 501

df = pd.DataFrame()

with open('./fgm_tables/tableProperties') as f:
    lines = f.readlines()

    data_lines = [x.strip() for x in lines[38:]]

    table_values = []
    for item in data_lines:
        if (is_number(item)):
            table_values.extend([float(item)])
        if (item == '('):
            table_values.pop(-1)
    pv_values = table_values[0:pv_scale]
    zeta_values = table_values[pv_scale:pv_scale + zeta_scale]
    f_values = table_values[pv_scale+zeta_scale:]

    zeta_cl = []
    for _ in range(pv_scale):
        for zeta in zeta_values:
            zeta_cl.extend(f_scale * [zeta])
    df['zeta'] = zeta_cl

    f_cl = []
    for _ in range(pv_scale):
        for _ in range(zeta_scale):
            f_cl.extend(f_values)
    df['f'] = f_cl

    pv_cl = []
    for pv in pv_values:
        pv_cl.extend(zeta_scale * f_scale * [pv])
    df['pv'] = pv_cl

# %%
# read data from table
tables = glob.glob('./fgm_tables/*_table')
for idx, fname in enumerate(tables):
    # if idx > 1:
        # break
    print(fname, idx + 1, '/', len(tables))

    with open(fname, 'r') as f:
        lines = f.readlines()

        data_lines = [x.strip() for x in lines[18:]]

        data_values = []
        for item in data_lines:
            # for item in data_lines[10:20]:
            s = re.split('[{}]', item)
            if (len(s) > 1):
                data_values.extend(int(s[0]) * [float(s[1])])
            if (len(s) == 1 and is_number(s[0])):
                data_values.extend([float(s[0])])
            if (s[0] == '('):
                data_values.pop(-1)

        assert (len(data_values) == pv_scale * zeta_scale * f_scale)
        # data_values=np.asarray(data_values)
        sp_name = fname.split('/')[2].split('_')[0]
        if not (sp_name in df.columns):
            df[sp_name] = data_values

df.to_hdf('tables_of_fgm.H5', key='of_tables', complib='zlib', complevel=9)
