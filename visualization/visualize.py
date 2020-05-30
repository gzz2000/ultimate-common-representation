import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt('similarity.txt')
a = np.flipud(a)

plt.imshow(a, cmap='magma')
plt.colorbar()


num = 6
plt.xticks(np.arange(0,num),np.arange(1,num+1))
plt.yticks(np.arange(0,num),np.arange(num,0,step=-1))

plt.xlabel('Network 2')
plt.ylabel('Network 1')


plt.show()
print(a)