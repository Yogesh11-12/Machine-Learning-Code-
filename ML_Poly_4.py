import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#import statsmodels.api as sm
data=pd.read_csv("FuelConsumptionCo2.csv",squeeze=True,usecols=["ENGINESIZE","CO2EMISSIONS"])
real_x=data.iloc[1:15,0].values
real_y=data.iloc[1:15,1].values
real_x=real_x.reshape(-1,1)
real_y=real_y.reshape(-1,1)
#training_x,testing_x,training_y,testing_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)
LR=LinearRegression()
LR.fit(real_x,real_y)
PR=PolynomialFeatures(degree=4)
PR_x=PR.fit_transform(real_x)
PR.fit(PR_x,real_y)
LR2=LinearRegression()
LR2.fit(PR_x,real_y)
plt.scatter(real_x,real_y, color='red')
plt.plot(real_x,LR2.predict(PR.fit_transform(real_x)))
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2ENMISSION")
plt.title("POLYNOMIAL_REGRESSION")
plt.show()



