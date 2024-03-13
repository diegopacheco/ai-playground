import pandas as pd
import numpy as np

dictionary = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}
df = pd.DataFrame(dictionary)
print(df)

sum = df.sum()
print(sum)

df = pd.DataFrame(pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]))
mean = df.mean()
print("Age mean " + str(mean))

df = pd.DataFrame(pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]))
max = df.max()
print("Age max " + str(max))

df = pd.DataFrame(pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]))
min = df.min()
print("Age min " + str(min))

