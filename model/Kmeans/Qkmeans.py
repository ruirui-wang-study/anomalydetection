import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
# import basicmetric
from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

X= pd.read_csv("./goodFeatures.csv")
y= pd.read_csv("./goodLabels.csv")
X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y)



import time

# 记录开始时间
start_time = time.time()

import numpy as np

# 初始化Q表，状态空间中每个状态对应动作空间大小的Q值列表
# 这里假设状态空间中有10个状态，动作空间为 {-1, 0, 1}，对应Q表为10x3的二维数组
num_states = 10
num_actions = 3
Q = np.zeros((num_states, num_actions))

# 定义状态空间和动作空间的取值范围
# state_space = np.linspace(0.1, 50, num_states)  # 状态空间范围从0.1到50均匀分布
state_space = np.arange(2, 11)  # K值范围从 2 到 10

action_space = np.array([-1, 0, 1])  # 动作空间为 {-1, 0, 1}

# 定义Q-learning参数
learning_rate = 0.1
discount_factor = 0.9
num_episodes = 100  # 迭代次数
exploration_rate=0.5
state = 5
# 迭代更新Q表
for episode in range(num_episodes):
    print("episode:",episode)
    # 初始化状态，这里假设状态初始值为5
    

    # 选择动作  这里应该是根据e- greedy策略选择动作而不是选择最优的
    # action_idx = np.argmax(Q[state_space == state, :])  # 根据当前状态选择最优动作的索引
    
    # action = action_space[action_idx]
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
    # 执行动作并观察奖励和新状态
    # 这里省略执行动作和观察奖励的过程，具体应根据问题实际情况实现
    # action = action_space[action_idx]

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
    # clf = svm.SVC(kernel='rbf',C=new_state,gamma=0.95)
    # precision,recall,f1,accuracy,false_alarm_rate=basicmetric.calMetrics(clf,X_train,y_train,X_test,y_test)
    # # scores = cross_val_score(clf, X, y, cv=5)
    # score=0.25*precision+0.25*recall+0.25*f1+0.25*accuracy+0.25*false_alarm_rate
    # reward = np.mean(score)
    kmeans = KMeans(n_clusters=new_state).fit(X_train)
    labels = kmeans.labels_
    # 计算内部评估指标
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_train, labels)
    calinski_harabasz = calinski_harabasz_score(X_train, labels)
    davies_bouldin = davies_bouldin_score(X_train, labels)

    print("Inertia:", inertia)
    print("Silhouette Score:", silhouette)
    print("Calinski-Harabasz Score:", calinski_harabasz)
    print("Davies-Bouldin Score:", davies_bouldin)
    reward = silhouette_score(X_train, labels)

    # 计算新状态下的最大Q值
    max_q_new_state = np.max(Q[new_state_idx, :])

    # 更新Q值
    state_index = np.where(state_space == state)[0][0]
    Q[state_index, action_index] += learning_rate * (reward + discount_factor * max_q_new_state - Q[state_index, action_index])
    print(state,action,Q[state_index, action_index])
    # 更新状态
    state = new_state

# 得到最优的惩罚参数C
best_k_idx = np.argmax(Q[:, :])
best_k = state_space[best_k_idx]
print("Best K:", best_k)


# 记录结束时间
end_time = time.time()

# 计算训练时间
training_time = end_time - start_time
print("训练时间：", training_time, "秒")