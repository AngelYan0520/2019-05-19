import pandas as pd

data = pd.read_csv('./house_price.csv')  # 读取house_price.csv
data1 = data.dropna()  # 删除空值
pd.set_option('display.max_columns', None)  # 最大输出
data2 = pd.get_dummies(data1[['dist', 'floor']])  # 处理成自变量
data3 = data2.drop(['dist_shijingshan', 'floor_high'], axis=1)  # 有选择的删除
data4 = pd.concat([data3, data1[['roomnum', 'halls', 'AREA', 'subway', 'school', 'price']]], axis=1)  # 将处理好的数据进行组合
print(data4)
x = data4.iloc[:, :-1]  # 除最后一列
y = data4.iloc[:, -1:]  # 第一列和最后一列
print(y)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
x_train, x_text, y_train, y_text = train_test_split(x, y, test_size=0.3, random_state=42)
model = linear_model.LinearRegression().fit(x_train, y_train)
request = model.predict([[0, 0, 0, 0, 0, 0, 0, 3, 2, 60, 1, 1]])
print(request)
score = model.score(x_text, y_text)
print(score)
print(model.coef_)  # 系数
print(model.intercept_)  # 节数


