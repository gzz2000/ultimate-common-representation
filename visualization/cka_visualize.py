import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dir = '../converter/untrained/'
a = np.loadtxt(dir + 'cka.txt')
a = np.flipud(a)
t= sns.heatmap(a, vmin=0.00, vmax=1.00, cmap='magma', xticklabels=list(range(1,a.shape[0] + 1)),yticklabels=list(range(a.shape[0],0,-1)))

plt.show()
