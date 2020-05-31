import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dir = '../converter/untrained/'
a = np.loadtxt(dir + 'cka.txt')
a = np.flipud(a)
t= sns.heatmap(a, vmin=0.00, vmax=1.00, cmap='magma', xticklabels=list(range(1,6)),yticklabels=list(range(5,0,-1)))

t.set(xlabel='Network B',ylabel='Network A')

plt.savefig('untrained_cka.png')
plt.show()
