import pandas as pd
from keras.models import load_model
import numpy as np
import time
from pre import read_csv_data

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# %%
x_train, y_train, df, in_scaler, out_scaler = read_csv_data('data')

ref = df.loc[df['p'] == 40]
x_test = in_scaler.transform(ref[['p', 'he']])

model = load_model('mayer.H5')

# %%
xs = np.concatenate([x_test for i in range(20)])
time_av = []
for i in range(10):
    st = time.time()
    predict_val = model.predict(xs.astype(float),batch_size=1024*8)
    time_cpu_infer = time.time() - st
    time_av.append(time_cpu_infer)
    # print(time_cpu_infer)
time_av = np.asarray(time_av)
print("The cpu inference time is ",time_av.mean())
