#SLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

data_set = pd.read_csv("C:/Users/bhakt/.spyder-py3/Salary_Data.csv")
print(data_set)

x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,1].values

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 10,train_size = 20, random_state=0)  

from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)

pred_train = regressor.predict(x_train)

mtp.scatter(x_train, y_train, color="green")   
mtp.plot(x_train, pred_train, color="red")    
mtp.title("Salary vs Experience (Training Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()   


#test
pred_test = regressor.predict(x_test)

#visualizing the Test set results  
mtp.scatter(x_test, y_test, color="blue")   
mtp.plot(x_train, pred_train, color="red")    
mtp.title("Salary vs Experience (Test Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()  

#MR
# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd
#importing datasets  
data_set= pd.read_csv('50_CompList.csv') 
#Extracting Independent and dependent Variable  
x= data_set.iloc[:, :-1].values  
y= data_set.iloc[:, 4].values  
#Catgorical data  
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
labelencoder_x= LabelEncoder()  
x[:, 3]= labelencoder_x.fit_transform(x[:,3])  
onehotencoder= OneHotEncoder(categorical_features= [3])    
x= onehotencoder.fit_transform(x).toarray()  
#avoiding the dummy variable trap:  
x = x[:, 1:]  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, rando
                                                 
#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)  
                                                   
#Predicting the Test set result;  
y_pred= regressor.predict(x_test)  
print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test))

