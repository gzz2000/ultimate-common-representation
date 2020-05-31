import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#a=np.loadtxt("H_1.txt")
#a = np.loadtxt('similarity_cka_adv_low.txt')
a = np.loadtxt('convert_ratio_cka_low.txt')
a = np.flipud(a)
t = sns.heatmap(a, vmin=0, vmax=0.9, cmap='PuBu', xticklabels=list(
    range(1, 7)), yticklabels=list(range(6, 0,-1)))
#t = sns.heatmap(a, vmin=0, vmax=10000, cmap='PuBu', xticklabels=list(
#    range(0, 10)), yticklabels=list(range(1, 7)))
t.set(xlabel='Network B', ylabel='Network A')
#t.set(xlabel='Label',ylabel='Layer')

plt.show()
