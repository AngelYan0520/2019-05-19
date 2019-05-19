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
plt.show()
x3 = data.iloc[:, :2]

