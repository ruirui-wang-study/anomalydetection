import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 随机森林排序前十的特征
X= pd.read_csv("goodFeatures.csv")
y= pd.read_csv("goodLabels.csv")
X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y)


# 使用SVM进行分类并输出热力图
svm = SVC(kernel='rbf', C=20.0, gamma='auto')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d",cmap='RdBu_r',center=300)
# # 设置图形标题和坐标轴标签
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# # 显示图形
plt.show()

# # 打印混淆矩阵
print(cm)


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# from sklearn.metrics import confusion_matrix

# # 读取数据
# X = pd.read_csv("goodFeatures.csv")
# y = pd.read_csv("goodLabels.csv")

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0, stratify=y)

# # # 使用PCA进行特征降维至2维
# # pca = PCA(n_components=2)
# # X_train_pca = pca.fit_transform(X_train)
# # X_test_pca = pca.transform(X_test)

# # 使用SVM进行分类
# svm = SVC(kernel='rbf', C=1.0, gamma='auto')
# svm.fit(X_train, y_train)
# y_pred = svm.predict(X_test_pca)

# # 绘制SVM模型的图
# plt.figure(figsize=(8, 6))

# # 绘制散点图
# plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train.values.ravel(), cmap=plt.cm.Paired, edgecolors='k', s=70)
# plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test.values.ravel(), cmap=plt.cm.Paired, marker='x', s=70)

# # 绘制决策边界
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()

# # 创建网格来评估模型
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = svm.decision_function(xy).reshape(XX.shape)

# # 绘制决策边界和支持向量
# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

# # 设置图例
# plt.legend(['Train', 'Test', 'Support Vectors'], loc='upper right')

# # 设置坐标轴标签
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')

# 显示图像
# plt.show()
