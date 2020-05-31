import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt('data.txt')
a = a.T

print(a)

convert_ratio = a[0]
train_loss = a[1]
test_loss = a[2]
train_acc = a[3]
test_acc= a[4]

x_data = np.arange(0,21)

'''
plt.plot(x_data,convert_ratio,color="red",linestyle = "--",marker='o',markersize=4)
plt.ylabel('Convert Ratio',fontsize=13)
plt.xlabel('epoch',fontsize=13)
ax=plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-0.5,20.5)
plt.ylim(0,1)
'''
'''
l1, = plt.plot(x_data,train_loss,color ="red",linestyle = "--",marker="o",markersize=4)
l2, = plt.plot(x_data,test_loss,color ="blue",linestyle = "--",marker="o",markersize=4)
plt.ylabel('Loss',fontsize=13)
plt.xlabel('epoch',fontsize=13)
ax=plt.gca()
plt.legend(handles=[l1,l2],labels=['train_loss','test_loss'])
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-0.5,20.5)


'''
l1, = plt.plot(x_data,train_acc,color ="red",linestyle = "--",marker="o",markersize=4)
l2, = plt.plot(x_data,test_acc,color ="blue",linestyle = "--",marker="o",markersize=4)
plt.ylabel('Accuracy',fontsize=13)
plt.xlabel('epoch',fontsize=13)
ax=plt.gca()
plt.legend(handles=[l1,l2],labels=['train_acc','test_acc'])
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-0.5,20.5)
plt.ylim(0,1)





plt.show()