# Arima-
In this I have used the time series data in which further it is predicting the upcoming data
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import acf , pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
df=pd.DataFrame(dataset)
#df.Date_time=pd.to_datetime(df.Date_time,format='%Y-%m-%d ')
#df.index=df.Date_time
df.head()
sale=df.Sales
acf_1= acf(sale) 
test_df=pd.DataFrame([acf_1]).T
test_df.columns=['Auto-correlation']
test_df.index +=1
test_df.plot(kind='bar')
plt.show()
pacf_1= pacf(sale) 
test_df=pd.DataFrame([pacf_1]).T
test_df.columns=['Partial-Autocorrelation']
test_df.index +=1
test_df.plot(kind='bar')
plt.show()
result= ts.adfuller(sale)
result
sale_diff=sale-sale.shift()
diff=sale_diff.dropna()
acf_1_diff=acf(diff)
test_df=pd.DataFrame([acf_1_diff]).T
test_df.columns=['First difference Auto-correlation']
test_df.index +=1
test_df.plot(kind='bar')
plt.show()
pacf_1_diff=pacf(diff)
test_df=pd.DataFrame([pacf_1_diff]).T
test_df.columns=['First difference Partial-Autocorrelation']
test_df.index +=1
test_df.plot(kind='bar')
plt.show()
sale_matrix=sale.as_matrix()
model=ARIMA(sale_matrix,order=(0,1,1))
model_fit=model.fit(disp=0)
print(model_fit.summary())
prediction=model_fit.predict(1,9,'Sales')
prediction
print(acf_1.reshape(-1,1))
print('\n')
print(pacf_1.reshape(-1,1))
#print(pacf_1)
print('\n')

predict_adjust=np.exp(prediction)
predict_adjust
plt.plot(predict_adjust)
plt.title('Forecast_Sale')
plt.show()
print(model_fit)
