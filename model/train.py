import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FeatureSelectionRL:
    def __init__(self, max_features=10, min_features=5, dataset_path="dataset.csv"):
        self.max_features = max_features
        self.min_features = min_features
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path)
        self.features = self.dataset.columns[:-1]  # 所有特征
        self.selected_features = set()
        self.env_reset()

    def env_reset(self):
        self.selected_features = set(self.features[:8])  # 初始化特征组

    def get_state(self):
        return len(self.selected_features), self.selected_features

    def get_reward(self):
        # 划分数据集
        X = self.dataset[list(self.selected_features)]
        y = self.dataset['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 训练分类器
        clf = SVC()
        clf.fit(X_train, y_train)

        # 预测并计算分类评估参数
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # 使用分类准确度作为奖励，因为我们的目标是得到分类效果最优的特征组
        return acc

    def step(self, action):
        if action == "add" and len(self.selected_features) < self.max_features:
            new_feature = set(self.features) - self.selected_features
            reward = 0
            for f in new_feature:
                self.selected_features.add(f)
                reward += self.get_reward()
                self.selected_features.remove(f)
            reward /= len(new_feature)
            self.selected_features.add(max(new_feature, key=lambda x: self.get_reward()))
        elif action == "remove" and len(self.selected_features) > self.min_features:
            reward = 0
            for f in self.selected_features:
                self.selected_features.remove(f)
                reward += self.get_reward()
                self.selected_features.add(f)
            reward /= len(self.selected_features)
            self.selected_features.remove(min(self.selected_features, key=lambda x: self.get_reward()))
        else:
            reward = self.get_reward()
        return reward

    def train(self, num_episodes=100):
        for episode in range(num_episodes):
            state = self.get_state()
            print(f"Episode {episode + 1}: Current feature size={state[0]}, Selected features={state[1]}")
            done = False
            while not done:
                action = self.get_action()
                reward = self.step(action)
                new_state = self.get_state()
                print(f"    Action={action}, Reward={reward:.4f}, New feature size={new_state[0]}, Selected features={new_state[1]}")
                done = (state == new_state)
                state = new_state

    def get_action(self):
        # 这里可以根据具体策略选择动作，这里简化为随机选择
        actions = ["add", "remove", "keep"]
        return actions[0]

if __name__ == "__main__":
    feature_selector = FeatureSelectionRL()
    feature_selector.train()
    print("Final selected features:", feature_selector.selected_features)
