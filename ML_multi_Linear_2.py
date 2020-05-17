import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
print(pred_y[3])
print(testing_y[3])
    


