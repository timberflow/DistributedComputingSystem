# 期末

## 内容描述:

寻找优化spark运行pipeline程序速度的部分代码级别的方法。

这里pipeline程序指深度学习或者机器学习里的算法，这一类模型的训练方法都是对每条数据做model的前向传播，然后反向传播计算梯度，最后把所有数据算得的梯度累加起来再使用梯度下降。

这个过程中数据的前向传播全部都使用窄依赖的算子，属于同一个stage，只有最后的累加部分会用到reduce之类的宽依赖算子。

本次实验就基于此做一个代码上的性能优化，包括RDD的持久化和宽依赖算子两个方面的优化。
