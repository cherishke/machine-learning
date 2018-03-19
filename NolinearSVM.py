#coding=utf-8
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 标准化
from sklearn.grid_search import GridSearchCV  #自动调参，适用于数据小的数据集
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

iris=datasets.load_iris()   #一些数据集
X = iris.data
y = iris.target
print iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.) # 分割训练集和测试集

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)   #数据标准化 StandardScaler().fit_transform(iris.data)公式为x'=（x-x(平均)）/s(方差)^2
X_test_std = scaler.transform(X_test)    #transform直接标准化

param_grid = {'C':[1e1,1e2,1e3, 5e3,1e4,5e4],
              'gamma':[0.0001,0.0008,0.0005,0.008,0.005,]}
clf = GridSearchCV(svm.SVC(kernel='rbf',class_weight='balanced'),param_grid,cv=10)  #rbf核函数，cv交叉验证次数
clf = clf.fit(X_train_std,y_train)
print clf.best_estimator_    #得出的最佳的SVC参数

print clf.score(X_test_std,y_test)
y_pred=clf.predict(X_test_std)   #预测结果

print(classification_report(y_test,y_pred,target_names=iris.target_names))
print(confusion_matrix(y_test,y_pred,labels=range(iris.target_names.shape[0])))  #纵坐标表示预测的是谁，横坐标表示标准的是谁。对角线的值越大，预测能力越好。