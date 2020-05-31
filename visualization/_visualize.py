import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

<<<<<<< HEAD
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
=======
dir = '../CKA/'
a = np.loadtxt(dir + 'converter_adv_cka.txt')
a = np.flipud(a)
t = sns.heatmap(a, vmin=0.00, vmax=1.00, cmap='magma', xticklabels=list(range(1,a.shape[0] + 1)),yticklabels=list(range(a.shape[0],0,-1)))
t.set(xlabel='Network B', ylabel='Network A')
>>>>>>> d223cae0548148ff87168142177a128178e6d970

plt.savefig('converter_adv_cka.png')
plt.show()
