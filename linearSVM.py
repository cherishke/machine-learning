#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC

np.random.seed(8)

# 线性可分：
array = np.random.randn(20,2)
X = np.r_[array-[3,3],array+[3,3]]
y = [0]*20+[1]*20
print X[0]
print X[20]
print y

clf=svm.SVC(kernel="linear") #创建分类器对象
clf.fit(X,y)    #用训练数据拟合分类器模型

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),   #取x中第一维或第二维最大最小值
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max)) #linspace默认以相同间隔，输出50个值；得到50*50的两个矩阵

# 得到向量w  : w_0x_1+w_1x_2+b=0
w = clf.coef_[0]   #得到w向量值
f = w[0]*xx1 + w[1]*xx2 + clf.intercept_[0]+1  # 加1后才可绘制 -1 的等高线 [-1,0,1] + 1 = [0,1,2]
plt.contour(xx1, xx2, f,[0,1,2],colors = 'r') # 绘制分隔超平面、H1、H2
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Paired)
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],color='k') # 绘制支持向量点
plt.show()
