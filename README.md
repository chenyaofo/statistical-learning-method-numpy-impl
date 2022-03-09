# *统计学习方法 (李航)* Numpy实现 / Numpy Implementaton for  *Statistical Learning Method (by Li Hang)*

本仓库旨在提供*统计学习方法*一书中算法的`numpy`实现。诚然在GitHub上已经有了许多类似的实现，但是本仓库有着以下优点：
This repo aims to provide numpy implementation for the algorithms in the book *Statistical Learning Method*.
I notice there are lots of implementtions in the GitHub. Here, I highlight the advantage of my implementations:

 - **快速运行**：可以直接在**Google Colab**上直接运行，无需下载或克隆代码到本地。Directly **running in Google Colab** without downloading/cloning the repo to the local.
 - **高效运行**：使用`numpy`中向量化实现替代python中的循环等操作。Implement by vectorized built-in functions in `numpy` instead of vanilla python operations, leading to better performance.
 - **真实数据集**：使用来自libsvm中的数据集而不是生成的数据。Using more practical datasets from libsvm instead of synthetic datasets.
 - **无需提前下载数据集**：数据集将以在线的方式直接加载无需额外手动下载。Loading datasets directly from the internet (libsvm website) without extra manual downloading.

这个仓库不是：This repo is not:

 - **算法详解**：只关注代码实现而不是算法本身的原理。details of algorithms. It focuses on implementations instead of other details of algorithms.
 - **通用算法库**：本实现仅仅用于快速理解算法，并未对算法进行封装，也未考虑实际使用中的各类情况。a general algorithm library. This implementation is only used to quickly understand the algorithm, it does not encapsulate the algorithm, and does not consider various situations in actual use.

## 监督学习 / Supervised Learning

 - K近邻算法 / *K*-nearest Neighbors Algorithm <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/KNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

 - 朴素贝叶斯法 / Naive Bayes Classifier <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/naive_bayes_classifier.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - 决策树
 / Decision Tree <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/decision_tree.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - 最大熵模型 / Maximum Entropy Model <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/maximum_entropy_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - 逻辑斯蒂回归 / Logistic Regression <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/logistic_regression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - 支持向量机 Supprt Vector Machine <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/support_vector_machine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - 提升方法 / AdaBoost <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/adaboost.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - EM算法 / Expectation Maximization Algorithm <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/EM.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - 隐马尔可夫模型 / Hidden Markov Model <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/hidden_markov_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

## 无监督学习 / Unsupervised Learning

 - K均值聚类 / K-means Clustering <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/K_means_clustering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - 层次聚类 / Hierachical Clustering <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/hierachical_clustering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - 主成分分析 / Principal Components Analysis <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/PCA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - 潜在语义分析 / Latent Semantic Analysis <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/latent_semantic_analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - 概率潜在语义分析 / Probabilistic Latent Semantic Analysis <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/probabilistic_latent_semantic_analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - 潜在狄利克雷分配 / Latent Dirichlet Allocation <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/latent_dirichlet_allocation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**

 - PageRank算法 / PageRank Algorithm <a href="https://colab.research.google.com/github/chenyaofo/statistical-learning-method-numpy-impl
/blob/master/algorithms/pagerank.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> **TODO**
