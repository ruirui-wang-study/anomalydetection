# 计算模型评估参数

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# 假设你有一个特征矩阵X和对应的标签y
# X是特征矩阵，每一行表示一个样本，每一列表示一个特征
# y是标签，表示每个样本的类别（0或1）

# 生成示例数据
X = np.random.rand(100, 10)
y = np.random.randint(2, size=100)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier()

def calMetrics(X_train,y_train,X_test,y_test,y_pred):
# 在训练集上训练分类器
    # clf.fit(X_train, y_train)

    # # 在测试集上进行预测
    # y_pred = clf.predict(X_test)

    # 计算性能指标
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    false_alarm_rate = fp / (fp + tn)

    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1-score:", f1)
    # print("Accuracy:", accuracy)
    # print("False Alarm Rate:", false_alarm_rate)
    return precision,recall,f1,accuracy,false_alarm_rate
