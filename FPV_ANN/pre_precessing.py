import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_hdf('./data/fpv_df.H5')

plt.scatter(df['PV'],df['f'],df['T'])
plt.show()