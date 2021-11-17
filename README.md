# Stanford CS231n 自学资料

- 持续更新（如果不偷懒）
- 已更新assignment1

## 课程

最近在项目大老板要求之下开始系统性学习计算机视觉知识。被强烈要求学习的内容中就包含斯坦福经典课程CS231n（李飞飞），同时学习的还有密歇根安娜堡的课程Deep Learning for Computer Vision

1. [CS231n](http://cs231n.stanford.edu)，经典计算机入门资料。YouTube上有课程录像。目前主流是2017年公开版本。
2. [Deep Learning for Computer Vision](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/). [YouTube](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) （YouTube上只有实时自动翻译，看着比较难受，B站上有机翻版本，但是翻译质量极差）

以上两门课都是高质量的计算机视觉（深度学习方面）的基础课程，同时需要做CS231n课后作业。两门课都是Fei Fei的博士生Johnson主讲，有一些重复内容可以跳过。

***

## 关于作业！！！

切身体会：初次接触这门课程，搞明白作业要求和形式***真的很麻烦***

- 2021版本[assignment 1](https://cs231n.github.io/assignments/2021/assignment1_colab.zip )压缩包(作业内容基本一致)
- [Cifar-10-python](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)数据集（可以自行下载解压到dataset路径，也可以执行路径下的相应配置文件自动下载）

### 作业的内容

以assignment 1为例，要求完成完成 kNN，SVM，Softmax分类器，以及一个两层的神经网络分类器的实现。原理参见课程**1-4节**内容。

### 作业的形式

以assignment 1 **kNN**为例，作业的形式类似于**填空**

- 首先，在下载的文件夹第一层中有很多**ipynb**文件（例如```knn.ipynb```)。对于不熟悉的同学而言可能会懵掉，但其实这是一个在**Jupyter**中运行的文件，它允许你分步执行代码，且在代码中间添加笔记说明等。**CS231n的题目要求就包含在各个.ipynb文件中**，我们需要阅读其中的说明去理解每一步所做的事情。
  ❗Jupyter在中文路径下无法正常运行，若用户（C盘user）含中文则会遇到该情况

- 接下来就是要完成.ipynb文件里要求的任务了。以```knn.ipynb```为例，其第一个任务出现在

  ```python
  # Open cs231n/classifiers/k_nearest_neighbor.py and implement
  # compute_distances_two_loops.
  
  # Test your implementation:
  dists = classifier.compute_distances_two_loops(X_test)
  print(dists.shape)
  ```

  我们查看该段代码的上方说明👇

  ```markdown
  First, open `cs231n/classifiers/k_nearest_neighbor.py` and implement the function `compute_distances_two_loops` that uses a (very inefficient) double loop over all pairs of (test, train) examples and computes the distance matrix one element at a time.
  ```

  任务：打开指定路径下的```k_nearest_neighbor.py```文件，完成其中的**compute_distances_two_loops**功能模块供调用
  
- 在**compute_distances_two_loops**中，我们看到***TODO***

  ```python
  for i in range(num_test):
              for j in range(num_train):
                  #####################################################################
                  # TODO:                                                             #
                  # Compute the l2 distance between the ith test point and the jth    #
                  # training point, and store the result in dists[i, j]. You should   #
                  # not use a loop over dimension, nor use np.linalg.norm().          #
                  #####################################################################
                  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  
                  pass
                 
                  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  ```

  我们需要做的就是在  ***pass***  的地方补全代码，实现要求的功能。例如，在此处的目标是通过两层循环实现L2距离的计算。
- 在所有地方完成填空之后，在Jupyter中逐步运行.ipynb文件（点击每个cell，按住Shift+Enter），即可得到相应结果。
  
### Tricks
1. 作业中的正向传播和反向传播基本由numpy矩阵计算实现。其中反向传播时，利用矩阵向量计算局部梯度较为抽象，建议可以先尝试在低维度自行计算单个数据点的情况（而非一批N个数据一起计算），再类推至一个batch的情况。
2. 作业整体是基于numpy实现的，如果对numpy的基本知识、操作和数据维度等不太熟悉（**尤其是维度，非常容易出问题**），建议首先学习numpy知识，并可以自行创建一个小的验证程序，随时对不确定的操作、维度等进行实例化输出验证。CS231n官方亦非常重视这一点，且提供了关于[numpy基本知识的介绍](https://cs231n.github.io/python-numpy-tutorial/)，可以首先学习。
