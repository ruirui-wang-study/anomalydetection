import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score

class QLearning:
    def __init__(self, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.5, num_iterations=500):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.num_iterations = num_iterations

        self.q_table = pd.DataFrame(columns=['C', 'Reward'])

    def choose_action(self, state):
        if np.random.uniform() < self.exploration_rate or state not in self.q_table.index:
            action = np.random.uniform(0.1, 10)  # randomly select C value
        else:
            state_action = self.q_table.loc[state, :]
            action = state_action['C'].idxmax()
        return action

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

            clf = SVC(C=action)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_val)
            pr = precision_score(y_val, y_pred)
            re = recall_score(y_val, y_pred)
            fs = f1_score(y_val, y_pred)
            ac = accuracy_score(y_val, y_pred)
            ba = balanced_accuracy_score(y_val, y_pred)

            reward = pr + re + fs + ac + ba
            print(reward)

            self.learn(str(X_train), action, reward)
import RF_feature
X,y=RF_feature.preprocess()

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize QLearning
ql = QLearning()

# Start training
ql.train(X_train, y_train, X_val, y_val)

# After training, you can get the best C value from Q-table
best_C = ql.q_table.idxmax()
print("Best C value:", best_C)
