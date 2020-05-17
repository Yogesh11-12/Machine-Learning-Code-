import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
data=pd.read_csv("FuelConsumptionCo2.csv",squeeze=True,usecols=["ENGINESIZE","CYLINDERS","CO2EMISSIONS"])
real_x=data.iloc[1:25,[0,1]].values
real_y=data.iloc[1:25,2].values
#real_x=real_x.reshape(-1,1)
#real_y=real_y.reshape(-1,1)
training_x,testing_x,training_y,testing_y=train_test_split(real_x,real_y,test_size=0.2,random_state=0)
KN=KNeighborsClassifier(n_neighbors=7, p=2)
KN.fit(training_x,training_y)
pred_y=KN.predict(testing_x)
print('prediction value: ',pred_y)
print('testing_y: ',testing_y)
x_set,y_set=training_x,training_y
X1,X2=np.meshgrid((np.arange(start=x_set[:,0].min()-1, stop=x_set[:,0].max()+1, step=0.01),(np.arange(start=x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step=0.01))
plt.contourf(X1,X2,KN.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
                   plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1]),
                   c=ListedColormap(('red','green'))(i),label=j)
plt.title("KNN")
plt.xlabel("ENGINESIZE")
plt.ylabel('CO2EMISSION')
plt.legend()                  
plt.show()                  
                                                                                          
                                                                                          
                   



