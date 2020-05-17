import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
data=pd.read_csv("FuelConsumptionCo2.csv",squeeze=True,usecols=["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","CO2EMISSIONS"])
real_x=data.iloc[:,0:3].values
real_y=data.iloc[:,3].values
#real_x=real_x.reshape(-1,1)
#real_y=real_y.reshape(-1,1)
training_x,testing_x,training_y,testing_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)
MLR=LinearRegression()
MLR.fit(training_x,training_y)
pred_y=MLR.predict(testing_x)
MLR.coef_
MLR.intercept_
real_x=np.append(arr=np.ones((1067,1)).astype(int),values=real_x,axis=1)
x_opt=real_x[:,[0,1,2,3]]
reg_OLS=sm.OLS(endog=real_y,exog=x_opt).fit()
print(reg_OLS.summary())
    


