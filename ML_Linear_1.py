import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv("FuelConsumptionCo2.csv",squeeze=True,usecols=["ENGINESIZE","CO2EMISSIONS"])
real_x=data.iloc[:,0].values
real_y=data.iloc[:,1].values
real_x=real_x.reshape(-1,1)
real_y=real_y.reshape(-1,1)
training_x,testing_x,training_y,testing_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)
lin=LinearRegression()
lin.fit(training_x,training_y)
pred_y=lin.predict(testing_x)
plt.scatter(testing_x,testing_y,color="green")
plt.plot(training_x,lin.predict(training_x),color="red")
plt.title("CO2EMISSIONOF CAR TESTING DATA")
plt.xlabel("ENGINE SIZE")
plt.ylabel("CO2EMISSION")
plt.show()
    


