import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# 读取数据
data = pd.read_csv('./dataset.csv')
# 对每种标签进行抽样,每种标签只留下20%的数据样本
data = data.groupby(' Label').apply(lambda group: group.sample(frac=0.1))

# 重置索引
data = data.reset_index(drop=True)
print(set(data[' Label']))
# data[' Label'] = data[' Label'].apply({
#             'DoS Slowhttptest':'Anormal',
#             'BENIGN':'Normal',
#             'DoS Hulk':'Anormal',
#             'Heartbleed':'Anormal',
#             'DoS GoldenEye':'Anormal',
#             'DoS slowloris':'Anormal'
#         }.get)

# 删除带有空值和无穷值的行
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# Standardize numerical columns
numerical_cols = data.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Encode non-numerical columns
categorical_cols = data.select_dtypes(exclude=[np.number]).columns
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# 随机森林进行特征提取，留下10个特征
X = data.drop([' Label'], axis=1)
y = data[' Label']
print(set(y))


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
selected_features = X.columns[indices][:10]
X = X[selected_features]
print(X.head())
print(X.shape)
X.to_csv("goodFeatures.csv", index=False)
y.to_csv("goodLabels.csv", index=False)