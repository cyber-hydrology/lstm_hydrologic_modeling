#!/usr/bin/env python
# coding: utf-8

# In[14]:


# suppress warnings
import warnings
warnings.simplefilter('ignore')


# In[15]:


# import required python libraries
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import scipy as sp
import tensorflow as tf
import pandas as pd
import statsmodels.api as sm
import sklearn as sk


# In[16]:


print(tf.__version__)


# In[121]:


# data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/WL_jungnang_2010_2020.csv')


# In[17]:


pwd


# In[19]:


data = pd.read_csv('/home/hchoi/Jamsu_lstm/Fulldata(11-17)-v1.csv')


# In[20]:


data.head()


# data의 변수명을 df로 변경

# In[21]:


df = data


# 불러온 시계열 자료를 가시화 하기

# In[23]:


plot_cols = ['StreamFlow','SeaLevel','Jamsubridge']
plot_features = df[plot_cols]
_ = plot_features.plot(subplots=True)


# 전체자료를 train, val, test 기간별로 분리하기

# In[24]:


df = df[plot_cols]

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
#train_df = df
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):]
#val_df = df[int(n*0.8):int(n*0.9)]
#test_df = df[int(n*0.9):]
test_df = df[int(n*0.7):]


num_features = df.shape[1]


# train 기간 자료의 평균 및 표준편차 계산하기

# In[25]:


train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


# In[26]:


df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')


# In[27]:


df.describe().transpose()


# In[28]:


df_std.tail()


# In[29]:


plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)


# In[33]:


# time step
ts = 48
# data is standardized
X_train = np.asarray([np.array([train_df.SeaLevel.values[i+j] for j in range(ts)])
                      for i in range(len(train_df.SeaLevel) - ts)]).reshape(-1,ts,1)
y_train = train_df.SeaLevel.values[ts:]
X_train.shape, y_train.shape


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


# LSTM
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

model_LSTM = Sequential()
model_LSTM.add(LSTM(10, input_shape=(ts, 1)))
model_LSTM.add(Dense(1, activation="linear"))
model_LSTM.compile(loss='mse', optimizer='adam')


# In[36]:


model_LSTM.summary()


# In[37]:


plt.plot(y_train[:], 'ro-', label="truth")
plt.plot(model_LSTM.predict(X_train[:, :, :]), 'bs-', label="prediction")
plt.legend()
plt.title("before training")
plt.show()


# In[38]:


import time
start_time = time.time()

history_LSTM = model_LSTM.fit(X_train, y_train, epochs=20,batch_size=32)

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)


# In[39]:


# plot loss of LSTM
plt.plot(history_LSTM.history["loss"])
plt.show()


# In[40]:


plt.plot(y_train[:], 'ro-', label="truth")
plt.plot(model_LSTM.predict(X_train[:, :, :]), 'bs-', label="prediction")
plt.legend()
plt.title("after training")
plt.show()


# In[41]:


plt.plot(y_train[:],model_LSTM.predict(X_train[:, :, :]),'ro')
plt.title("after training")
plt.xlabel('observed')
plt.ylabel('simulated')
plt.show()


# In[42]:


# time step
ts = 24
# data is standardized
X_val = np.asarray([np.array([val_df.SeaLevel.values[i+j] for j in range(ts)])
                      for i in range(len(val_df.SeaLevel) - ts)]).reshape(-1,ts,1)


# In[43]:


y_val = val_df.SeaLevel.values[ts:]
X_val.shape, y_val.shape


# In[44]:


plt.plot(y_val[:], 'ro-', label="truth")
plt.plot(model_LSTM.predict(X_val[:, :, :]), 'bs-', label="prediction(val)")
plt.legend()
plt.title("validation with sigmoid")
plt.show()


# In[45]:


plt.plot(y_val[:],model_LSTM.predict(X_val[:, :, :]),'ro')
plt.title("validation")
plt.xlabel('observed')
plt.ylabel('simulated')
plt.show()


# In[46]:


print(max(y_val[:]))
print(np.argmax(y_val))
ymax_i = np.argmax(y_val)


# In[47]:


print(model_LSTM.predict(X_val[:, :, :]).shape)


# In[48]:


y_predict = model_LSTM.predict(X_val[:, :, :])


# In[49]:


print(y_predict.shape)


# In[50]:


plt.plot(y_val[ymax_i-48:ymax_i+48], 'ro-', label="truth")
plt.plot(y_predict[ymax_i-48:ymax_i+48,:], 'bs-', label="prediction(val)")
plt.legend()
plt.show()


# In[51]:


# from the normalized values to the real values
X_train_real = ( model_LSTM.predict(X_train[:, :, :]) + train_mean[0] ) * train_std[0]
y_train_real = (y_train + train_mean[0] ) * train_std[0]

plt.plot(y_train_real[:], 'ro-', label="truth")
plt.plot(X_train_real[:], 'bs-', label="prediction")
plt.legend()
plt.title("after training")
plt.show()


# In[52]:


# from the normalized values to the real values
X_val_real = ( model_LSTM.predict(X_val[:, :, :]) + train_mean[0] ) * train_std[0]
y_val_real = (y_val + train_mean[0] ) * train_std[0]

plt.plot(y_val_real[:], 'ro-', label="truth")
plt.plot(X_val_real[:], 'bs-', label="prediction")
plt.legend()
plt.title("test period")
plt.show()


# In[54]:


#!pip install hydroeval


# In[55]:


from sklearn.metrics import mean_squared_error
import hydroeval as he
import os


# In[56]:


he.evaluator(he.kge, X_train_real[:,0], y_train_real)


# In[57]:


# train period
# root mean square error
rmse_train = (mean_squared_error(y_train_real,X_train_real[:,0]))**0.5
rmse_val  = (mean_squared_error(y_val_real, X_val_real[:,0]))**0.5
nse_train = he.evaluator(he.nse, X_train_real[:,0], y_train_real)
nse_val = he.evaluator(he.nse, X_val_real[:,0], y_val_real)
print('48-hr memory, 10 single layers,32 batch,200 epochs, linear act')
print('RMSE(train):' , "{:.3f}".format(rmse_train), 'm')
print('RMSE(val)  :'  , "{:.3f}".format(rmse_val), 'm')
print('NSE(train) :' , "{:.3f}".format(nse_train[0]))
print('NSE(val)   :' , "{:.3f}".format(nse_val[0]))


# In[58]:


plt.plot(y_val_real[ymax_i-48:ymax_i+48], 'ro', label="observed")
plt.plot(X_val_real[ymax_i-48:ymax_i+48], 'b-', label="simulated")
plt.legend()
plt.xlabel("Time(hour)")
plt.ylabel("Water level(m)")
plt.title("48-hr memory, 10 single layers,32 batch,200 epochs, linear act")
plt.show()


# In[59]:


plt.plot(y_val_real,X_val_real,'o')
plt.title("validation(12-hr memory)")
plt.xlabel('Observed')
plt.ylabel('Simulated')
plt.show()

