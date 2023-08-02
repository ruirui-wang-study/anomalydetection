import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据集，假设数据集文件名为data.csv
data = pd.read_csv('dataset.csv')

# 统计每个标签的数量
label_counts = data[' Label'].value_counts()

# 设置样式
# sns.set_style('whitegrid')  # 使用seaborn美化样式
plt.figure(figsize=(10, 6))  # 设置图的大小

# 绘制柱状图
sns.barplot(x=label_counts.index, y=label_counts.values,label='count', linewidth=0.1)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Label Distribution')

# 旋转x轴标签，以免重叠
plt.xticks(rotation=45, ha='right')

# 显示每个柱子上的数量值
for i, v in enumerate(label_counts.values):
    plt.text(i, v + 10, str(v), ha='center', va='center', fontweight='bold', color='black')
# 添加图例
plt.legend(loc="upper right")
# plt.tight_layout()  # 自动调整图的布局

plt.show()
