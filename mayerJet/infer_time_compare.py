import pandas as pd
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import numpy as np
import time
from pre import read_csv_data
import os

# %%
# load data
x_train, y_train, df, in_scaler, out_scaler = read_csv_data('data')

ref = df.loc[df['p'] == 40]
x_test = in_scaler.transform(ref[['p', 'he']])
xs = np.concatenate([x_test for i in range(20)])

# %%
# CPU inference
with tf.device('/cpu:0'):
    model = load_model('mayer.H5')

    time_av = []
    for i in range(50):
        st = time.time()
        predict_val = model.predict(xs.astype(float), batch_size=1024 * 8)
        time_cpu_infer = time.time() - st
        time_av.append(time_cpu_infer)
        # print(time_cpu_infer)
    time_av = np.asarray(time_av)
print("The cpu inference time is ", time_av.mean())

# %%
# GPU inference
with tf.device('/gpu:0'):
    model = load_model('mayer.H5')
    bs = 1024 * 4
    time_batch = []
    time_tot = []
    for i in range(500):
        st = time.time()
        _ = model.predict(xs[:bs].astype(float), batch_size=bs)
        time_gpu_infer_batch = time.time() - st
        time_batch.append(time_gpu_infer_batch)

        st = time.time()
        _ = model.predict(xs.astype(float), batch_size=bs)
        time_gpu_infer = time.time() - st
        time_tot.append(time_gpu_infer)

    time_batch = np.asarray(time_batch)
    time_tot = np.asarray(time_tot)
    print('there are ', round(xs.shape[0] / bs + 0.5), 'batches')
    print('Batch inference time is ', time_batch.mean())
    print('sequential inference time is ', time_batch.mean() * round(xs.shape[0] / bs + 0.5))
    print('Gpu inference time is ', time_tot.mean())

# %%
# release gpu memory
from numba import cuda

cuda.select_device(0)
cuda.close()
