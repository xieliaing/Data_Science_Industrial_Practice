# coding: utf-8
#=====================================================================
# # 1-LinearRegression
#=====================================================================
import pandas as pd
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.model_selection import train_test_split

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = boston.target
# 数据集划分, 训练集：验证集 = 7:3
x_train, x_val, y_train, y_val = train_test_split(X, Y, 
                                                  test_size=0.3, 
                                                  random_state=20)
linear = linear_model.LinearRegression() # 建立最小二乘法线性模型
linear.fit(x_train, y_train) #拟合模型
print(linear.score(x_train, y_train)) # 返回线性拟合的R方
print(linear.coef_) # 获得各个变量的权重
print(linear.intercept_) # 获得各个变量的偏置项


# # 查看权重参数
res_parms = pd.DataFrame([float("%.5f"%i) for i in linear.coef_], 
                          index=x_train.columns, 
                          columns=['params'])
res_parms.sort_values(by=['params'], inplace=True, ascending=True)
print(res_parms)

#=====================================================================
# # 2-logistic regression
#=====================================================================
import pandas
from sklearn import linear_model
# 此处需要先下载kaggle原始数据，然后执行本文件夹中的预处理文件11.1_titanic_preprocessing.py
# 数据加载
data = pd.read_csv('train_dp.csv')
Y = data['survived']
X = data.drop(columns=['survived'])

# 数据集划分
x_train, x_val, y_train, y_val = train_test_split(X, Y, 
                                                  test_size=0.3, 
                                                  random_state=20)

lr = linear_model.LogisticRegression(penalty='l2') # 建立最小二乘法线性模型
lr.fit(x_train, y_train) #拟合模型
print(lr.score(x_train, y_train)) # 返回线性拟合的R方
print(lr.coef_)# 获得各个变量的权重
print(lr.intercept_) # 获得各个变量的偏置项


# # 查看权重参数
res_parms = pd.DataFrame([float("%.5f"%i) for i in lr.coef_[0]], index=x_train.columns, columns=['params'])
res_parms.sort_values(by=['params'], inplace=True, ascending=True)
res_parms

#=====================================================================
# # 3-decision tree
#=====================================================================
import pandas as pd
from sklearn import tree
# 此处需要先下载kaggle原始数据，然后执行本文件夹中的预处理文件11.1_titanic_preprocessing.py
# 数据加载
data = pd.read_csv('train_dp.csv')
Y = data['survived']
X = data.drop(columns=['survived'])

# 数据集划分
x_train, x_val, y_train, y_val = train_test_split(X, Y, 
                                                  test_size=0.3, 
                                                  random_state=20)
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(x_train, y_train) # 模型训练
y_pred = clf.predict(x_val) #模型预测
from sklearn.metrics import precision_score, recall_score, accuracy_score
precision_score(y_val, y_pred), recall_score(y_val, y_pred), accuracy_score(y_val, y_pred)

# 树模型可视化

import matplotlib.image as mpimg
import os
import pydotplus
from IPython.core.display import Image
from sklearn.externals.six import StringIO
import graphviz
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=x_train.columns,
                                 class_names=True, filled=True, rounded=True, proportion=True,
                                 special_characters=True, node_ids=True, )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png('rules_of_tree.png')
show_pic = 1
if show_pic == True:
    lena = mpimg.imread('rules_of_tree.png')
    plt.imshow(lena)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.savefig('a.png')
    plt.show()
    print('decision tree end')

#=====================================================================
# # 4-knn
#=====================================================================
import pandas as pd
from sklearn import neighbors
# 此处需要先下载kaggle原始数据，然后执行本文件夹中的预处理文件11.1_titanic_preprocessing.py
# 数据加载
data = pd.read_csv('train_dp.csv')
Y = data['survived']
X = data.drop(columns=['survived'])

# 数据集划分
x_train, x_val, y_train, y_val = train_test_split(X, Y, 
                                                  test_size=0.3, 
                                                  random_state=20)
n_neighbors = 15
knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
knn.fit(x_train, y_train)

# 查看不同K值对评估结果的影响
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
dic = {}
k_list = np.arange(5, 50, 10)
for k in k_list:
    knn = neighbors.KNeighborsClassifier(k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_val)
    dic["k="+str(k)] = [  recall_score(y_val, y_pred), 
                          precision_score(y_val, y_pred), 
                          accuracy_score(y_val, y_pred), 
                          f1_score(y_val, y_pred)]

df = pd.DataFrame(dic, index=['precision', 'recall', 'accuracy','f1'])
plt.plot(df)
plt.legend(labels = df.columns)
plt.grid(axis='both')

#=====================================================================
# # 5-贝叶斯
#=====================================================================
import pandas as pd
from sklearn.naive_bayes import GaussianNB
# 此处需要先下载kaggle原始数据，然后执行本文件夹中的预处理文件11.1_titanic_preprocessing.py
# 数据加载
data = pd.read_csv('train_dp.csv')
Y = data['survived']
X = data.drop(columns=['survived'])

# 数据集划分
x_train, x_val, y_train, y_val = train_test_split(X, Y, 
                                                  test_size=0.3, 
                                                  random_state=20)
n_neighbors = 15
bayes = GaussianNB(var_smoothing=0)
bayes.fit(x_train, y_train)
print(bayes.theta_) # 获得各个属性属于该类的均值
print(bayes.sigma_) # 获得各个属性属于该类的方差
print(bayes.predict_proba(x_val))
print(bayes.predict(x_val))