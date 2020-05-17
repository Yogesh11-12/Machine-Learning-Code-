import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
df=pd.read_csv('C:/Users/yogesh yadav/Downloads/loan_train.csv')
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
real_x=df[['age','Principal','terms','age','Gender']].values
scaler=StandardScaler()
real_x=scaler.fit_transform(real_x)
real_x=np.array(real_x)
print(real_x.shape)
df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'],value=[0,1],inplace=True)
real_y=df['loan_status'].values
real_y=np.array(real_y).reshape(-1,1).ravel()
print(real_y.shape)
train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)
DR=DecisionTreeRegressor(random_state=0)
DR.fit(train_x,train_y)
pred_y=DR.predict(test_x)
#X_grid=np.array(min(test_x),max(test_x),0.01)
#X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(real_x[:,0],real_y,color='red')
plt.plot(test_x[:,0],pred_y)
plt.xlabel("Person Detail")
plt.ylabel("Loan_status")
plt.title("DECISION_TREE_Algorithm")
plt.show()



