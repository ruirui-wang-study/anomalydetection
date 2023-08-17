import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pylab import mpl
import seaborn as sns


# # 设置颜色代码
# color1 = "#038355" # 孔雀绿
color1 = "#0066CC" # lan
color2 = "#FF0033" # hong

plt.style.use('seaborn-whitegrid')
plt.grid(linestyle='dashed')

#mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.sans-serif'] = ['Times New Roman']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

t=pd.read_csv('time.log').values
t_dpacc=pd.read_csv('time_dpacc.log').values
y=np.arange(0,30)

plt.xlabel("Duration(s)")
plt.ylabel("RTT(s)")
l1,=plt.plot(y,t,linestyle='--',markeredgecolor='black',linewidth=1,color=color1,markersize=3,marker='o')
l2,=plt.plot(y,t_dpacc,linestyle='--',markeredgecolor='black',linewidth=1,color=color2,markersize=3,marker='^')
# l3,=plt.plot(y,t_5,linestyle='--',markeredgecolor='black',linewidth=1,color=color3,markersize=3,marker='*')
legend=plt.legend([l1,l2],['DPACC disabled', 'DPACC enabled'],frameon=True,loc='best')
legend.get_frame().set_facecolor('white')
frame = legend.get_frame()
frame.set_edgecolor('lightgray')
# 设置坐标轴样式
for spine in plt.gca().spines.values():
    spine.set_edgecolor("black")
plt.show()