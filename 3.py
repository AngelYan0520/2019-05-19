import pandas as pd

data = pd.read_csv('./house_price1.csv')
x1 = data.iloc[:, :1]
x2 = data[['area']]
y = data.iloc[:, -1:]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
axes = Axes3D(fig)
axes.scatter3D(x1, x2, y)

x3 = data.iloc[:, :2]
from sklearn import linear_model
model = linear_model.LinearRegression().fit(x3, y)
print(model.coef_)
print(model.intercept_)
import numpy as np
X1, X2 = np.meshgrid(x1, x2)
Y = 0.695186 * X1 + 0.01225383 * X2 + 0.15032823
axes.plot_surface(X1, X2, Y, alpha=0.3)
plt.show()

