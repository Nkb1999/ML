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
