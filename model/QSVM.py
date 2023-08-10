# 你想使用Q-learning来调整SVM的C参数。下面是使用 Q-learning 来调整C参数的基本思路：

# 初始化Q表。Q表的行应该表示C参数的可能值，列表示可能的动作（在这种情况下，可能的动作是：加一，减一，不变）。

# 对每个训练周期：

# 初始化状态，即设定C的初始值。
# 在该周期的每一步，根据Q表选择一个动作，并根据所选动作更新C的值。然后，训练一个新的SVM模型。
# 计算reward，这可能是SVM分类器在验证集上的性能度量（如准确性、召回率、F1得分等）。
# 更新Q表：Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
# 更新当前状态为新状态。
# 在所有训练周期结束后，得到的Q表可以用于选择最优的C参数。

# 这是一个非常基础的实现，你可能需要对其进行调整以更好地适应你的特定问题。例如，你可能需要调整学习率和奖励函数，或者添加一个ε-greedy策略来平衡探索和开发。你也可能需要设置一个更适合你问题的C参数的范围。

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

FORWARD = 0
BACKWARD = 1
class QLearning:
    # def __init__(self, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.5, num_iterations=500):
    #     self.learning_rate = learning_rate
    #     self.discount_factor = discount_factor
    #     self.exploration_rate = exploration_rate
    #     self.num_iterations = num_iterations

    #     self.q_table = pd.DataFrame(columns=['C', 'Reward'])
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.9):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((len(state_space), len(action_space)))

    def choose_action(self, state):
        if np.random.uniform() < self.exploration_rate or state not in self.q_table.index:
            action = np.random.uniform(0.1, 10)  # randomly select C value
        else:
            state_action = self.q_table.loc[state, :]
            action = state_action['C'].idxmax()
        return action
    
    def take_action(self,action):
        if action == 1:
            self.new_state_idx = np.argmin(np.abs(state_space - state)) + 1  # 下一个状态的索引
        elif action == -1:
            self.new_state_idx = np.argmin(np.abs(state_space - state)) - 1  # 上一个状态的索引
        else:
            self.new_state_idx = np.argmin(np.abs(state_space - state))  # 保持当前状态的索引
        clf = svm.SVC(kernel='rbf',C=new_state,gamma=0.95)
        precision,recall,f1,accuracy,false_alarm_rate=basicmetric.calMetrics(clf,X_train,y_train,X_test,y_test)
        # scores = cross_val_score(clf, X, y, cv=5)
        score=0.25*precision+0.25*recall+0.25*f1+0.25*accuracy+0.25*false_alarm_rate
        reward = np.mean(score)
        
    def get_next_action(self, state):
        if random.random() > self.exploration_rate: # Explore (gamble) or exploit (greedy)
            return self.greedy_action(state)
        else:
            return self.random_action()
    def greedy_action(self, state):
        # Is FORWARD reward is bigger?
        if self.q_table[FORWARD][state] > self.q_table[BACKWARD][state]:
            return FORWARD
        # Is BACKWARD reward is bigger?
        elif self.q_table[BACKWARD][state] > self.q_table[FORWARD][state]:
            return BACKWARD
        # Rewards are equal, take random action
        return FORWARD if random.random() < 0.5 else BACKWARD
    def random_action(self):
        return FORWARD if random.random() < 0.5 else BACKWARD
    
    
    def update_state(self):
        # 确保新状态索引不超出边界
        self.new_state_idx = max(0, min(new_state_idx, num_states - 1))
        # 获取新状态
        self.new_state = state_space[new_state_idx]
        
    def update(self, old_state, new_state, action, reward):
        # 更新Q表
        # 这里假设在执行动作后，得到了奖励 reward
        

        # 计算新状态下的最大Q值
        max_q_new_state = np.max(Q[new_state_idx, :])

        # 更新Q值
        Q[state_space == state, action_idx] += learning_rate * (reward + discount_factor * max_q_new_state - Q[state_space == state, action_idx])
        
        
        # # Old Q-table value
        # old_value = self.q_table[action][old_state]
        # # What would be our best next action?
        # future_action = self.greedy_action(new_state)
        # # What is reward for the best next action?
        # future_reward = self.q_table[future_action][new_state]

        # # Main Q-table updating algorithm
        # new_value = old_value + self.learning_rate * (reward + self.discount * future_reward - old_value)
        # self.q_table[action][old_state] = new_valuex

        # Finally shift our exploration_rate toward zero (less gambling)
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta

    def learn(self, state, action, reward):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0],
                    index=self.q_table.columns,
                    name=state,
                )
            )
        q_predict = self.q_table.loc[state, action]
        q_target = reward + self.discount_factor * self.q_table.loc[state, :].max()
        self.q_table.loc[state, action] += self.learning_rate * (q_target - q_predict)

    def train(self, X_train, y_train, X_val, y_val):
        for i in range(self.num_iterations):
            action = self.choose_action(str(X_train))

            

            self.learn(str(X_train), action, reward)

X= pd.read_csv("./goodFeatures.csv")
y= pd.read_csv("./goodLabels.csv")
X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y)


# C=1.0
# state_space=C
# action_space=[1,-1,0]

# # Initialize QLearning
# ql = QLearning(state_space,action_space)

# # Start training
# ql.train(X_train, y_train, X_test, y_test)

# # After training, you can get the best C value from Q-table
# best_C = ql.q_table.idxmax()
# print("Best C value:", best_C)
import time

# 记录开始时间
start_time = time.time()

import numpy as np

# 初始化Q表，状态空间中每个状态对应动作空间大小的Q值列表
# 这里假设状态空间中有10个状态，动作空间为 {-1, 0, 1}，对应Q表为10x3的二维数组
num_states = 10
num_actions = 3
Q = np.random.rand(num_states, num_actions)

# 定义状态空间和动作空间的取值范围
state_space = np.linspace(0.1, 50, num_states)  # 状态空间范围从0.1到50均匀分布
action_space = np.array([-1, 0, 1])  # 动作空间为 {-1, 0, 1}

# 定义Q-learning参数
learning_rate = 0.1
discount_factor = 0.9
num_episodes = 100  # 迭代次数
exploration_rate=0.5
state = 0.1
# 迭代更新Q表
for episode in range(num_episodes):
    print("episode:",episode)
    # 初始化状态，这里假设状态初始值为0.1
    

    # 选择动作  这里应该是根据e- greedy策略选择动作而不是选择最优的
    # action_idx = np.argmax(Q[state_space == state, :])  # 根据当前状态选择最优动作的索引
    
    # action = action_space[action_idx]
    if np.random.uniform() < exploration_rate or state not in Q.index:
        action_idx = np.random.uniform(1, 10)-1  # randomly select C value
    else:
        action_idx = np.argmax(Q[state_space == state, :])
    # 执行动作并观察奖励和新状态
    # 这里省略执行动作和观察奖励的过程，具体应根据问题实际情况实现
    action = action_space[action_idx]

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
    precision,recall,f1,accuracy,false_alarm_rate=basicmetric.calMetrics(clf,X_train,y_train,X_test,y_test)
    # scores = cross_val_score(clf, X, y, cv=5)
    score=0.25*precision+0.25*recall+0.25*f1+0.25*accuracy+0.25*false_alarm_rate
    reward = np.mean(score)

    # 计算新状态下的最大Q值
    max_q_new_state = np.max(Q[new_state_idx, :])

    # 更新Q值
    Q[state_space == state, action_idx] += learning_rate * (reward + discount_factor * max_q_new_state - Q[state_space == state, action_idx])
    print(state,action,Q[state_space == state, action_idx])
    # 更新状态
    state = new_state

# 得到最优的惩罚参数C
best_c_idx = np.argmax(Q[:, :])
best_c = state_space[best_c_idx]
print("Best C:", best_c)


# 记录结束时间
end_time = time.time()

# 计算训练时间
training_time = end_time - start_time
print("训练时间：", training_time, "秒")