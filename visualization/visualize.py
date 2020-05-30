import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

a = np.loadtxt('similarity_UU.txt')
a = np.flipud(a)
t= sns.heatmap(a, vmin=0.03, vmax=0.65, cmap='magma', xticklabels=list(range(1,7)),yticklabels=list(range(6,0,-1)))

t.set(xlabel='Network B',ylabel='Network A')

plt.show()
