# 期末

## 内容描述:

寻找优化spark运行pipeline程序速度的部分代码级别的方法。

这里pipeline程序指深度学习或者机器学习里的算法，这一类模型的训练方法都是对每条数据做model的前向传播，然后反向传播计算梯度，最后把所有数据算得的梯度累加起来再使用梯度下降。

这个过程中数据的前向传播全部都使用窄依赖的算子，属于同一个stage，只有最后的累加部分会用到reduce之类的宽依赖算子。

本次实验就基于此做一个代码上的性能优化，包括RDD的持久化和宽依赖算子两个方面的优化。另外只有这两个方面的优化教材上有提到。

这些结论已经被很多人证实过了，这次的实验主要在于验证以及比较。

## 使用方式:

代码实际上使根据spark logistic regression的框架改的，要运行可以通过test文件来运行。

也可以通过MLPRunner里的main函数来运行，但这种方式要改shell输入的参数，EditConfiguration里增加argument： 

```
./input ./output/weights "4 10 3"
```

最后面三个参数中中间的数是隐藏层大小，可以随意调整。层数不能改，我只实现了两层的。

集群上先上传jar到某个位置，其次最好先把输入文件放到hdfs上，我放在了hdfs根目录的spark_input/iris里，输出选择一个本地的路径。例如可以执行以下：

```
./spark-2.4.7/bin/spark-submit --master spark://localhost:7077 --class DSPPCode.spark.MLP.question.MLPRunner /home/ubuntu/myApp/MLP.jar hdfs://localhost:9000/user/ubuntu/spark_input/iris ~/output/weights "4 10 3"
```

