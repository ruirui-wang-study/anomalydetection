import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
import basicmetric
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
X= pd.read_csv("goodFeatures.csv")
y= pd.read_csv("goodLabels.csv")
X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y)
from pso_svm import utils
# from utils import data_handle_v1, data_handle_v2
# from config.config import args, kernel, data_src, data_path
# X_train, X_test, y_train, y_test = utils.data_handle_v1("D:/private/anomalydetection/model/pso_svm/data/Statlog_heart_Data.csv")

import time

# 记录开始时间
start_time = time.time()

import numpy as np
import math
# result = math.e ** math.pi
# 初始化Q表，状态空间中每个状态对应动作空间大小的Q值列表
# 这里假设状态空间中有10个状态，动作空间为 {-1, 0, 1}，对应Q表为10x3的二维数组
num_states = 10
num_actions = 3
Q = np.zeros((num_states, num_actions))

# 定义状态空间和动作空间的取值范围
state_space = np.linspace(0.1, 50, num_states)  # 状态空间范围从0.1到50均匀分布
action_space = np.array([-1, 0, 1])  # 动作空间为 {-1, 0, 1}

# 定义Q-learning参数
learning_rate = 0.1
discount_factor = 0.9
num_episodes = 10  # 迭代次数
exploration_rate=0.5
state = state_space[4]

y_pred_best=[]
best_C=0.1
bestModel = None
# 迭代更新Q表
for episode in range(num_episodes):
    # print("episode:",episode)
    for step in range(10):
        if np.random.uniform(0,1) < exploration_rate or state not in Q:
            # action_idx = np.random.randint(1, 10)-1  # randomly select C value
            action = np.random.choice(action_space)
            action_index = np.where(action_space == action)[0][0]
        else:
            # action_idx = np.argmax(Q[state_space == state, :])
            # 执行利用操作：选择Q值最大的动作
            state_index = np.where(state_space == state)[0][0]
            action_index = np.argmax(Q[state_index, :])
            action = action_space[action_index]

        # 更新状态
        if action == 1:
            new_state_idx = np.argmin(np.abs(state_space - state)) + 1  # 下一个状态的索引
        elif action == -1:
            new_state_idx = np.argmin(np.abs(state_space - state)) - 1  # 上一个状态的索引
        else:
            new_state_idx = np.argmin(np.abs(state_space - state))  # 保持当前状态的索引

        # 确保新状态索引不超出边界
        new_state_idx = max(0, min(new_state_idx, num_states - 1))

        # 获取新状态
        new_state = state_space[new_state_idx]


        # 更新Q表
        # 这里假设在执行动作后，得到了奖励 reward
        clf = svm.SVC(kernel='rbf',C=new_state,gamma=0.95)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # if episode==num_episodes and step==10:
        #     cm = confusion_matrix(y_test, y_pred)
        #     sns.heatmap(cm, annot=True, fmt="d",cmap='RdBu_r',center=300)
        #     # # 设置图形标题和坐标轴标签
        #     plt.title('Confusion Matrix')
        #     plt.xlabel('Predicted Label')
        #     plt.ylabel('True Label')
        #     # # 显示图形
        #     plt.show()
        #     plt.savefig("qsvm_hotmap.png")
        precision,recall,f1,accuracy,false_alarm_rate=basicmetric.calMetrics(X_train,y_train,X_test,y_test,y_pred)
        # scores = cross_val_score(clf, X, y, cv=5)
        score=0.2*precision+0.2*recall+0.2*f1+0.2*accuracy+0.2*math.e **false_alarm_rate
        reward = np.mean(score)
        

        # 计算新状态下的最大Q值
        max_q_new_state = np.max(Q[new_state_idx, :])

        # 更新Q值
        state_index = np.where(state_space == state)[0][0]
        Q[state_index, action_index] += learning_rate * (reward + discount_factor * max_q_new_state - Q[state_index, action_index])
        if Q[state_index, action_index]==np.argmax(Q):
            y_pred_best=y_pred
            best_C=state
            bestModel=clf
        # print(state,action,Q[state_index, action_index])
        # 更新状态
        state = new_state

# 得到最优的惩罚参数C
best_q_idx = np.unravel_index(np.argmax(Q), Q.shape)
best_c = state_space[best_q_idx[0]]

# best_c = state_space[best_c_idx]
print("Best C:", best_c)
# cm = confusion_matrix(y_test, y_pred_best)
# sns.heatmap(cm, annot=True, fmt="d",cmap='RdBu_r',center=300)
# # # 设置图形标题和坐标轴标签
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# # # 显示图形
# plt.show()
# # plt.savefig("qsvm_hotmap.png")

# 记录结束时间
end_time = time.time()

# 计算训练时间
training_time = end_time - start_time
print("训练时间：", training_time, "秒")
print(Q)