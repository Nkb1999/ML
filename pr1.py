import pandas as pd
#1. Read the data from Data.csv to DataFrame “dataset”.
#dataset = pd.read_csv("C:/Users/bhakt/.spyder-py3/Data.csv", header=3)
dataset = pd.read_csv("C:/Users/bhakt/.spyder-py3/Data.csv")
print(dataset)

# 2. Display total number of rows and columns of dataset.
print("Total rows " , len(dataset), "\nTotal columns ", len(dataset.columns))

# 3. Display first 5 record from dataset.
result = dataset.head(5)
print(result)
 
print("=================================================")
# 4. Display all values of column “City” by accessing column name as attribute of dataset.
print(dataset.Country)

# =============================================================================

print("=================================================")
# 5. Display all values of column “City” by using indexing ( [ ] ) operator.
print(dataset['Country'])


print("=================================================")
# 6. Display “City” value from the first record.
print(dataset['Country'][0])


print("=================================================")
# 7. Display first record using iloc( ).
print(dataset.iloc[0])

print("=================================================")
# 8. Display all values of first column using iloc( ).
print(dataset.iloc[:,0])

print("=================================================")
# 9. Display “City” value from 2nd and 3rd records using iloc( ).
print(dataset.iloc[2:4,0])

print("=================================================")
# 10. Display “City” value from 2nd and 4th records using iloc( ).
print(dataset.iloc[2:5,0])

print("=================================================")
# 11. Display last 3 records from the dataset using iloc( ).
print(dataset.iloc[-3:])

print("=================================================")
# 12. Display all records using loc( ) where “City” value is “Bardoli”.
print(dataset.loc[dataset.Country=="Bardoli"])

print("=================================================")
# 13. Display all “City” values with number of records having that value using groupby( ).
print(dataset.groupby('Country').Country.count())

print("=================================================")
# 14. Display all records where “Age” is NULL.
print(dataset[pd.isnull(dataset.Age)])

print("=================================================")
# 15. Delete all rows with missing values.
print(dataset.dropna(axis=0,inplace=True))
print(dataset)

print("=================================================")
# 16. Delete all columns with missing values.
print(dataset.dropna(axis=1,inplace=True))
print(dataset)

print("=================================================")
# 17. Replace missing categorical values with most frequent values.
# =============================================================================
# value = dataset["Salary"].mode()[0]
# dataset["Salary"].fillna(value, inplace = True)
# print(dataset)
# =============================================================================

print("=================================================")
print("=================================================")

# 18. Replace missing “City” name with value “Bardoli”, if any.
print(dataset)
print("=================================================")
dataset.fillna("Bardoli", inplace = True)
print(dataset)
print("=================================================")

# =============================================================================
df = pd.read_csv('data.csv')

x = df["Calories"].mode()[0]

df["Calories"].fillna(x, inplace = True)
#=========================================================================

#Set "Duration" = 45 in row 7:
df.loc[7, 'Duration'] = 45

df.drop_duplicates(inplace = True)

-----------------------------------------------------------
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
#---------------------------------
#MLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds=pd.read_csv("Invest2Profit.csv")
df=pd.DataFrame(ds)

x=df.iloc[:,:-1].values
y=df.iloc[:,4].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# labelencode=LabelEncoder()
# x[:,3]=labelencode.fit_transform(x[:,3])
# hot=OneHotEncoder(categorical_features=4)
# x=hot.fit_transform(x).toarray()
ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[3])],remainder="passthrough")
x=np.array(ct.fit_transform(x))
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

pred_test=regressor.predict(x_test)
pred_train=regressor.predict(x_train)

print("Train Score:",regressor.score(x_train,y_train))
print("Test Score:",regressor.score(x_test,y_test))
#------------------------------------------------------------------------


