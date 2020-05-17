import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#import statsmodels.api as sm
data=pd.read_csv("FuelConsumptionCo2.csv",squeeze=True,usecols=["ENGINESIZE","CYCLINDERS","CO2EMISSIONS"])
real_x=data.iloc[1:15,[0,1]].values
real_y=data.iloc[1:15,2].values
real_x=real_x.reshape(-1,1)
real_y=real_y.reshape(-1,1)
#training_x,testing_x,training_y,testing_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)
RF=RandomForestRegressor(n_estimators=800,random_state=0)
RF.fit(real_x,real_y.ravel())
pred_y=RF.predict(real_x)
X_grid=np.arange(min(real_x),max(real_x),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(real_x,real_y, color='red')
plt.plot(X_grid,RF.predict(X_grid))
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2ENMISSION")
plt.title("RANDOM_FOREST")
plt.show()



