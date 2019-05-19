import pandas as pd

data = pd.read_csv('./house_price1.csv')
x = data[['area']]
y = data.iloc[:, -1:]

from sklearn import linear_model
model = linear_model.LinearRegression().fit(x, y)
a = model.coef_
b = model.intercept_

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='r')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')

# y = ax + b
c = a*x + b
plt.plot(x, c)
plt.show()

