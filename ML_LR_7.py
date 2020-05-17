import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#import statsmodels.api as sm
data=pd.read_csv("FuelConsumptionCo2.csv",squeeze=True,usecols=["ENGINESIZE","CYLINDERS","CO2EMISSIONS"])
real_x=data.iloc[1:25,[0,1]].values
real_y=data.iloc[1:25,2].values
#real_x=real_x.reshape(-1,1)
#real_y=real_y.reshape(-1,1)
training_x,testing_x,training_y,testing_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)
LR=LogisticRegression()
LR.fit(training_x,training_y)
pred_y=LR.predict(testing_x)
print('prediction value: ',pred_y)
print('testing_y: ',testing_y)



