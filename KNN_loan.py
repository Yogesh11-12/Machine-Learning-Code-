import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
df=pd.read_csv('C:/Users/yogesh yadav/Downloads/loan_train.csv')
#print(df.head())
df['due_date']=pd.to_datetime(df['due_date'])
df['effective_date']=pd.to_datetime(df['effective_date'])
#print(df['effective_date'])
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
real_x=df[['age','Principal','terms','age','Gender']].values
scaler=StandardScaler()
real_x=scaler.fit_transform(real_x)
real_x=np.array(real_x)
print(real_x[:,0])
df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'],value=[0,1],inplace=True)
real_y=df['loan_status'].values
real_y=np.array(real_y).reshape(-1,1).ravel()
#print(real_y)
train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.25,random_state=0)
knn=KNeighborsClassifier(n_neighbors=7,p=2)
knn.fit(train_x,train_y)
y_pred=knn.predict(test_x)
cm=confusion_matrix(test_y,y_pred)
#print(y_pred)
print("Confusion matrix: ",cm)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])
X,Y=train_x,train_y
h=.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = Y.min() - 1, Y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
plt.pcolormesh(xx, yy, y_pred, cmap=cmap_light)
plt.scatter(real_x[:,0],real_y,label='stars',color='green',marker='*',s=30)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % (n_neighbors))
plt.show()










