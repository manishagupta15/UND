#!/usr/bin/env python
# coding: utf-8

# In[87]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# import seaborn as sns


# In[88]:


df_path1=('SearchQueriesUS.csv')
df_raw=pd.read_csv(df_path1,index_col='Date')
df_raw.shape


# In[89]:


print(df_raw.describe())
print(df_raw.info())
plt.figure(figsize=(12,5))
plt.title('Daily Search Queries',fontsize = 20)
plt.ylabel('Total Search Queries',fontsize = 20)
plt.xlabel('Date',fontsize = 20)
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(df_raw['UnitedStates'])


# In[62]:


# df_raw.plot(figsize=(12,6))
# df_raw['Actual']=df_raw['UnitedStates'].rolling(7).sum()
# df=df_raw[['Actual']]
# df=df.iloc[7:]
# df.head(7)


# In[90]:


df=df_raw
df.head()


# In[91]:


train = df.iloc[:1674]
test = df.iloc[1674:]
# train = df.iloc[:1700]
# test = df.iloc[1700:]


# In[86]:


scaled_train


# In[92]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# df.head(),df.tail()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
# scaled_train[:10]
from keras.preprocessing.sequence import TimeseriesGenerator
# n_input = 7

# generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
# X,y = generator[0]
# print(f'Given the Array: \n{X.flatten()}')
# print(f'Predict this y: \n {y}')
# We do the same thing, but now instead for 12 months
n_input = 4
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit(generator,epochs=50)
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)


# In[93]:


last_train_batch = scaled_train[-4:]
last_train_batch = last_train_batch.reshape((1, n_input, n_features))
model.predict(last_train_batch)
scaled_test[0]
test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 
    
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[95]:


test.head(84)
# test_predictions


# In[97]:


true_predictions = scaler.inverse_transform(test_predictions)

test['Prediction'] = true_predictions
test.plot(figsize=(14,5))
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(test['Prediction'],test['UnitedStates']))
print(rmse)


# In[99]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
test.head(100)


# In[103]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(test['UnitedStates'],test['Prediction']))
print(rmse)

