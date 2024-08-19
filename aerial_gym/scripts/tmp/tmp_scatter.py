import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成示例三维轨迹数据：x, y, z坐标和速度值
num_points = 1000
x = np.random.rand(num_points)
y = np.random.rand(num_points)
z = np.random.rand(num_points)
speed = np.random.rand(num_points)  # 假设速度范围在0到1之间

# 绘制连续热力图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用plot_trisurf绘制热力图
surf = ax.plot_trisurf(x, y, z, cmap='hot', linewidth=0.2, facecolors=plt.cm.hot(speed))
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Continuous Heatmap of Speed')
plt.savefig(f'/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/charts_output/tmp_scatter_plot.png')
plt.show()