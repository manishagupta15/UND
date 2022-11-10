#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import joblib
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_process import arma_generate_sample
from sklearn.metrics import mean_squared_error
# ACF/PACF
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AR,ARResults
from pandas.plotting import autocorrelation_plot
#from statsmodels.tsa.ma_model import MA,MAResults
from itertools import product 
from tqdm import tqdm_notebook
from statsmodels.tsa.stattools import adfuller
from numpy import log
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
import os
# from dateutil.easter import easter
from matplotlib import dates as mpl_dates
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import adfuller
from numpy import log
from statsmodels.tsa.filters.hp_filter import hpfilter
from matplotlib.pyplot import figure
#Python Option setting
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#Python Option setting
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import pandas as pd
#import cs
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import NMF
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud as wc

from time import time
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope


###[['Canada','CA'],['UnitedStates','US'],['Germany','DE'],['Australia','AU'],['France','FRA'],['UnitedKingdom','GB']]


# modeltype='add'
# periods_forecast=180
i=0

ci=1.28 #80% Confidence Interval
ci_arima=0.2 #80% Confidence Interval
ci_prophet=0.95
#seasonal_mode='additive' #'multiplicative' #
seasonal_mode='multiplicative' #'multiplicative' #
changepoint_prior=0.05
if seasonal_mode=='multiplicative': 
    changepoint_prior=0.05
CustomTrendChangepoint =[]



# In[22]:


# ######pip install holidays==0.9.12
#####pip install pandas==0.25.3.
print(pd.__version__)





#Filter out Daily data
def callMarket(marketname,df_raw):
    df = df_raw[[marketname]]
    return df

def plotresult(test,predictions):
    title = 'Daily prediction'
    ylabel='DSQ'
    xlabel=''
    ax = test.plot(legend=True,figsize=(12,6),title=title)
    predictions.plot(legend=True)
    ax.autoscale(axis='x',tight=True)
    ax.set(xlabel=xlabel, ylabel=ylabel);
    
def mape(df): 
    actual, pred = np.array(df['Actual']), np.array(df['Predicted'])
    return np.mean(np.abs((actual - pred) / actual)) * 100

def resultset(test,predictions):
    resultdf = pd.concat([test, predictions], axis=1)
    resultdf.columns=['Actual','Predicted']
    resultdf['Error%']= (resultdf['Actual']-resultdf['Predicted'])/resultdf['Actual']
    print(resultdf.head(10));
    print("MAPE %:",np.mean(np.abs(resultdf['Error%']))*100)
    print("Max Error%: ", np.abs(resultdf['Error%']).max()*100)
    plotresult(test,predictions)
    return resultset

def DickeyFullerTest(df):
    result = adfuller(df)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1]> 0.05:
        print("Dont reject the null hypothesis. Series is Non-Stationary")
    else:
        print("Reject the null hypothesis. Series is Stationary")


def getSmoothdata(df_raw,MVType):
    if MVType =='R1':
	    df_r7= df_raw.rolling(window=7).mean() 
        # df_r7=df_r7.reset_index()
        # df_r7=df_r7.loc[7:]
        # df_r7=df_r7.set_index('Date')  
        #df_r7= df_raw[['United States','United Kingdom', 'Germany','Australia','Canada','France']].rolling(window=7).mean()
    else:
        df_r7= df_raw 
    df_r7=df_r7.reset_index()
    df_r7=df_r7.loc[0:]
    df_r7=df_r7.set_index('Date')
    return df_r7


###NEW
def prophet_model(modeltype,yearly_seasonality,holidays,changepoints_new,best_params):
    best_params=best_params
    
    import pandas as pd
    from fbprophet import Prophet
    if modeltype =='add':
        print("Additive")
        m = Prophet()
   
    elif modeltype =='mul':
        print("MULITPLICATIVE")
        m = Prophet(interval_width=0.95,yearly_seasonality=True,seasonality_mode= 'multiplicative',
        changepoint_prior_scale=0.05,holidays=holidays,
        seasonality_prior_scale=10,holidays_prior_scale=10,changepoint_range=0.8)

    elif modeltype =='mul_changepoint':
        print("MULITPLICATIVE")
        m = Prophet(interval_width=0.95,yearly_seasonality=True,seasonality_mode= 'multiplicative',
        changepoint_prior_scale=0.05,holidays=holidays,
        seasonality_prior_scale=10,holidays_prior_scale=10,changepoints=changepoints_new)
    
    elif modeltype=='custom':
        # 'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative', 'changepoint_range': 0.95, 'yearly_seasonality': 12
        print("custom")
        print(best_params)
        m = Prophet(growth='linear',
        interval_width=0.95,
        yearly_seasonality=best_params['yearly_seasonality'],
        seasonality_mode= best_params['seasonality_mode'],
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        holidays=holidays,
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        changepoint_range=best_params['changepoint_range'])

    elif modeltype=='custom_changepoint':
        print('custom_changepoint')
        m= Prophet(interval_width=0.95,
        yearly_seasonality=best_params['yearly_seasonality'],
        seasonality_mode= best_params['seasonality_mode'],
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        holidays=holidays,
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        changepoints=changepoints_new
                       )

    else: 
        print("mul_manual")
        m = Prophet(interval_width=ci_prophet,
        yearly_seasonality=12,
        seasonality_mode= 'multiplicative',
        changepoint_prior_scale=0.5,
        holidays=holidays, 
        holidays_prior_scale=10,
        seasonality_prior_scale= 2,
        changepoint_range=0.90)
    return m;



def check_mean_std(ts, name,marketname):

    rolmean = ts.rolling(window=7).mean()
    rolstd = ts.rolling(window=7).std()
    plt.figure(figsize=(12,8))   
    print(name)
    
    plt.style.use('seaborn')
    plt.figure(figsize=(12,8))
    plt.plot(ts, color='red',label='Original')
    plt.plot(rolmean, color='black', label='Rolling Mean')
    plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.gcf().autofmt_xdate()
    date_format = mpl_dates.DateFormatter('%d, %b, %Y')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.tight_layout()
    plt.xlabel("Date")
    plt.ylabel("DSQ")
    plt.title(' Rolling Mean & Standard Deviation : '+str(marketname[0][0]))
    plt.legend()
    plt.show()



def train_test_split(SmoothDf,split_date,marketname,Truncate_date):
    i=0
    Truncate_date=pd.to_datetime(Truncate_date)
    SmoothDf=SmoothDf[SmoothDf.index >= Truncate_date ]
    split_date = pd.to_datetime(split_date)
    df_temp=callMarket(marketname[i][0],SmoothDf)
    train= df_temp[df_temp.index <= split_date]
    test= df_temp[df_temp.index > split_date]
    train=train.reset_index()
    test=test.reset_index()
    train_sub = train.rename(columns={marketname[i][0]: 'y','Date': 'ds'})
    test_sub = test.rename(columns={marketname[i][0]: 'y','Date': 'ds'})
    return train,test,train_sub,test_sub


def get_CrossValidationResult(model,initial):
    from fbprophet import Prophet
    from fbprophet.plot import add_changepoints_to_plot
    from fbprophet.diagnostics import cross_validation
    from fbprophet.diagnostics import performance_metrics
    from fbprophet.plot import plot_cross_validation_metric
    df_cv = cross_validation(model,
                            horizon='70 days',
                            period='30 days',
                            # initial='1491 days'
                            initial=initial
                        )
    df_p = performance_metrics(df_cv)
    df_p.head()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1);


    fig = plot_cross_validation_metric(df_cv, metric='mape')
    fig.set_size_inches(20,5)
    plt.show();
    MinMAPE=df_p['mape'].min()
    MaxMAPE = df_p['mape'].max()

    return MinMAPE,MaxMAPE,df_cv,df_p



def showChangepoint(model,forecast):
    from fbprophet.plot import add_changepoints_to_plot
    fig = model.plot(forecast)
    add_changepoints_to_plot(fig.gca(), model, forecast)
    plt.show()

def showModelComponent(model,forecast):
    fig2 = model.plot_components(forecast)
    plt.show()



def cosol_rs(forecast,df_temp):
    i=0
    forecast_sub=forecast[['ds','yhat']]
    forecast_sub['ds']=forecast_sub['ds'].astype(str)
    forecast_sub['YhatR28sum']=forecast['yhat'].rolling(window=28).sum()
    forecast_sub['country']=marketname[i][1]   
    forecast_sub['YhatR7sum']=forecast['yhat'].rolling(window=7).sum()     
    df_temp_plot= df_temp
    df_temp_plot=df_temp_plot.reset_index()
    df_temp_plot=df_temp_plot.rename(columns={marketname[i][0]: 'Actual','Date': 'ds'})
    df_temp_plot['ds']=df_temp_plot['ds'].astype(str)
    Consol_df = forecast_sub.merge(df_temp_plot,on=['ds'],how='left')
    Consol_df['ActualR28sum']=Consol_df['Actual'].rolling(window=28).sum()
    Consol_df['ActualR7sum']=Consol_df['Actual'].rolling(window=7).sum()
    Consol_df['R28_APE']=abs(Consol_df['YhatR28sum']-Consol_df['ActualR28sum'])/Consol_df['ActualR28sum']
    Consol_df['is_Month_end']=Consol_df['ds'].apply(lambda x :pd.to_datetime(x).is_month_end)
    try:
        Consol_df['Daily_APE']=abs(Consol_df['yhat']-Consol_df['Actual'])/Consol_df['Actual']
    except:
        pass
    
    ####----- Ploting R7, R28

    plot_df=Consol_df.set_index('ds')
    plot_df[['yhat','Actual']].plot(figsize=(30, 5))
    plt.title(str(marketname[i][0])+':R1', fontsize=14)
    plot_df[['YhatR7sum','ActualR7sum']].plot(figsize=(30, 5))
    plt.title(str(marketname[i][0])+':R7', fontsize=14)
    plot_df[['YhatR28sum','ActualR28sum']].plot(figsize=(30, 5))
    plt.title(str(marketname[i][0])+':R28', fontsize=14)
    return Consol_df
	
	
	
import numpy as np

def set_changepoints(df, n_changepoints=25, changepoint_range=.95):
    df = df.sort_values('ds').reset_index(drop=True)
    hist_size = int(np.floor(df.shape[0] * changepoint_range))
    if n_changepoints + 1 > hist_size:
        n_changepoints = hist_size - 1
        print('n_changepoints greater than number of '+
              'observations. Using {}.'.format(n_changepoints))
    if n_changepoints > 0:
        cp_indexes = (np.linspace(0,
                                  hist_size - 1,
                                  n_changepoints + 1).
                      round().astype(np.int))
        changepoints = df.iloc[cp_indexes]['ds'].tail(-1)
    else:
        # set empty changepoints
        changepoints = pd.Series(pd.to_datetime([]), name='ds')
    return changepoints


def GetHolidays(Additional_holidays,marketname,Truncate_date,df_raw,split_date):
    from fbprophet.make_holidays import make_holidays_df
    MVType='R11'
    # split_date='2022-01-31'
    SmoothDf=getSmoothdata(df_raw,MVType)
    train,test,train_sub,test_sub=train_test_split(SmoothDf,split_date,marketname,Truncate_date)
    # train_sub
    year_list = pd.to_datetime(train_sub['ds']).dt.year.unique().tolist()
    # Identify the final year, as an integer, and increase it by 1
    year_list.append(year_list[-1] + 1)
    holidays = make_holidays_df(year_list=year_list,
                            country=marketname[0][1])
    # holidays;
    holidays_new=pd.DataFrame()

    holidays_new = pd.concat([holidays, Additional_holidays]
                    ).sort_values('ds').reset_index(drop=True)
    return holidays_new

# 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'multiplicative', 'changepoint_range': 0.8





df_path1=('SearchQueriesUS.csv')
df_raw=pd.read_csv(df_path1,index_col='Date')
# df_path1 = adlsFileSystemClient.open(inputfile)
# df=pd.read_csv(df_path1, delimiter=",")

df_raw=df
df_raw = df_raw.astype({marketname[i][0]: float}) #####----------------------------------------------------------------------------------------------------------------------------------------------- Change require
df_raw['Date'] = pd.to_datetime(df_raw['Date']).dt.date
df_raw=df_raw.sort_values(by='Date').reset_index(drop=True)
df_raw=df_raw.set_index('Date')
# df_raw=df_raw.reset_index(drop=True)
df_raw.info()
df_raw.head()
df_raw.plot(figsize=(30, 5))
plt.show()

df_raw=df_raw[[marketname[i][0]]]####-----------------------------------------------------------------------------------------------------------------------------------------------------------------------change

#### Plotting Rolling Mean and Standard Deviation of Datase
check_mean_std(df_raw[[marketname[i][0]]],'\n\nDSQ',marketname)


###STL Decompose & Anamoly Detection in historical data

i=0
df_decompose = df_raw[[marketname[i][0]]] #########--------------------------------------------------------------------------------------------------------------- Mexico

# fig, ax = plt.subplots()
# x = result.resid.index
# y = result.resid.values
# ax.plot_date(x, y, color='black',linestyle='--')

# # ax.annotate('Anomaly', (mdates.date2num(x[280]), y[280]), xytext=(20, 8), 
# #            textcoords='offset points', color='red',arrowprops=dict(facecolor='red',arrowstyle='fancy'))

# fig.autofmt_xdate()
# plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
plt.rc('figure',figsize=(20,8))
plt.rc('font',size=15)

result = seasonal_decompose(df_decompose ,model='multiplicative',freq=7)
fig = result.plot()

plt.rc('figure',figsize=(20,6))
plt.rc('font',size=15)


#### Anomaly
import math
df_stl_resid= result.resid
resid_mu = result.resid.mean()
resid_dev = result.resid.std()
# anything outside lower and upper limit is anamoly
lower = resid_mu - 3*resid_dev
upper = resid_mu + 3*resid_dev


lowerlimit=lower
upperlimit=upper
df_stl_resid['lowerlimit']=lowerlimit
df_stl_resid['upperlimit']=upperlimit
df_stl_resid.plot()


###### Anamoly Detection
from datetime import datetime
anomalies = df_decompose[(result.resid < lower) | (result.resid > upper)]
plt.figure(figsize=(20,5))
plt.plot(df_decompose)
for year in range(2017,2023):
    plt.axvline(datetime(year,1,1), color='k', linestyle='--', alpha=0.5)
    
plt.scatter(anomalies.index, anomalies.iloc[:,[0]], color='r', marker='D')
anomalies.dropna()

estimated = result.trend + result.seasonal
plt.figure(figsize=(12,4))
plt.plot(df_decompose)
plt.plot(estimated)

#Training and test split to check the error metrics
# df_r7_prep=df_r7.set_index('Date')






# In[23]:


Truncate_date=pd.to_datetime('2018-01-01')
split_date = '2022-07-31'


# **#Neural Prophet****

# In[25]:


def train_test_split(SmoothDf,split_date,Truncate_date):
    i=0
    Truncate_date=pd.to_datetime(Truncate_date)
    SmoothDf=SmoothDf[SmoothDf.index >= Truncate_date ]
    split_date = pd.to_datetime(split_date)
    # df_temp=callMarket(marketname[i][0],SmoothDf)
    train= SmoothDf[SmoothDf.index <= split_date]
    test= SmoothDf[SmoothDf.index > split_date]
    train=train.reset_index()
    test=test.reset_index()
    train_sub = train.rename(columns={marketname[i][0]: 'y','Date': 'ds'})
    test_sub = test.rename(columns={marketname[i][0]: 'y','Date': 'ds'})
    return train,test,train_sub,test_sub


# In[26]:


from neuralprophet import NeuralProphet
m = NeuralProphet(
    n_changepoints=50,
    yearly_seasonality=6,
    weekly_seasonality=True,
    changepoints_range=0.95,
    seasonality_mode='multiplicative',
    num_hidden_layers=5,
    trend_reg=3,
    learning_rate=0.01,
    ar_reg=0.1,
    n_forecasts=180,
    seasonality_reg=1
    #, n_lags=366
)
split_date = pd.to_datetime('2022-07-31')
Truncate_date = pd.to_datetime('2018-01-01')

train,test,train_sub,test_sub=train_test_split(df_raw,split_date,Truncate_date)

# create the data df with events


# history_df = m.create_df_with_events(train_sub, events_df)

metrics = m.fit(train_sub,freq="D")
future = m.make_future_dataframe(df=train_sub, periods=83)
forecast = m.predict(df=future)

fig_forecast = m.plot(forecast)

m.plot(forecast)
m.plot_components(forecast)
m.plot_parameters()


# In[29]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
forecast


# In[62]:


forecast = m.predict(train_sub)
fig = m.plot(forecast)
m.plot_parameters()

