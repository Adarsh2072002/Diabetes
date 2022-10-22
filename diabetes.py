

import pandas as pd
 
from sklearn.neighbors import KNeighborsClassifier       

from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv('diabetes.csv')
print(df.head())

x= df.iloc[:,:-1]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(x_train,y_train)

# pred=model.predict([[90,5]])
# print(pred)
pickle.dump(model,open('diabetes.pkl','wb'))