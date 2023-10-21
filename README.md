# **Project：向量数据库**
## **任务定义**
输入：大规模向量库 *G*，待查询的 *N* 个向量 {q1,q2,…,qN}

查询过程：遍历 *N* 个待查询向量，对于每个查询向量 qi(i = 1, 2, . . . N )，给出图像向量库中最相似的 *K*

个图像向量 Si={qi1,qi2,…,qiK}

输出：输出 *N* 个向量的最相似的 *K* 个向量的集合 {S1,S2,…,SN}

您所需要做的是设计查询的算法，完成待查询向量的**快速 准确**的查询。
## **数据集描述**
我们采用 A、B 数据集的方式测试查询算法的优越性，其中 A 数据集将会给出下载通道，方便调试算法。

B 数据集将不会提供下载链接，在提交完代码后，由机器计算性能得分，A/B 数据集参数如下： 

A：待查询向量数 *N* = 500，大规模向量库 *G* 中有 50 万条向量，向量维度为 512，*K* 为 50。

B：待查询向量数 *N* = 5000，大规模向量库 *G* 中有 500 万条向量，向量维度为 512，*K* 为 50。

A 数据集下载链接，https://pan.baidu.com/s/1jKLpwpE1vVodaDTsq2WL7A?pwd=tgi7, code: tgi7 
## **评价指标**
为了衡量算法的优越性，我们从准确性和速度两个方向进行评价。

准确率：对于每条查询 qi* (*i* = 1*,* 2*, . . . N* )，我们采用 *P* @50 评价查询结果的准确率。

速度：利用多条查询的平均时间 <i>t<sub>q</sub></i> 评价查询算法的速度。

成绩排名规则：在保障平均每条查询时间控制在200ms以内的前提下，按照准确率指标排名。如果平均查询时间超过200ms，则不进入排名。
## **组队名单**
组队名单是参照第一次作业的成绩进行组队，尽量做到每组实力均衡。


|第1组|胡辰恺|吴剑昊|刘奕骁|韩恺荣|徐泽宇||
| :- | :- | :- | :- | :- | :- | :- |
|第2组|邹鹤鸣|胡诗雅|夏恩博|王成章|沈轩甜||
|第3组|应周骏|张子豪|李予谦|孙宇桐|刘涵嫣|王子儒|
|第4组|王佳灿|刘紫雯|董昕鹏|王虹懿|包越|李昌骏|
|第5组|朱骏驰|钟好|路翔云|庄毅非|师野|谷佳兴|
## **提交说明**
1. B数据集格式与A数据集一样，只是数量规模更大，需要提供readme文档介绍如何编译/运行程序，方便助教测试性能。
1. 代码编程语言不限，测试环境有多核CPU和GPU硬件，可支持并发查询。
1. 数据预处理和索引创建时间不计入查询耗时。
1. 代码可多次提交，提交之后助教会尽快测试并告知查询时间与准确率的指标
1. 代码提交测试的截止日期是11月10号，之后助教不再接收代码测试。11月12号前需要在系统提交代码、算法文档（word）。
