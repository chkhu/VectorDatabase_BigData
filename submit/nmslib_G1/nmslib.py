# 50条查询所用时间 2.6622s
# 准确率99.80%


# 导入nmslib库和numpy库
import nmslib
import numpy as np
import time
import pickle


def build_index(xb):
    # 创建一个空的hnsw索引对象，并设置索引的参数
    index = nmslib.init(method='hnsw', space='l2')  # 使用欧氏距离（l2）作为相似度度量
    index_params = {'M': 16, 'efConstruction': 100,'post': 2}  # 设置构建索引时的参数，M是每个节点的最大连接数，efConstruction是搜索质量参数，越大越准确，但是速度越慢，post是后处理参数，表示在构建索引后要执行的局部优化步骤的数量
    # 将向量库添加到索引中，并保存索引到文件（可选）
    index.addDataPointBatch(xb)  # 假设xb是向量库，是一个形状为(500000, d)的numpy数组，500000是向量库的大小
    index.createIndex(index_params, print_progress=True)  # 创建索引，并打印进度信息
    index.saveIndex('hnsw_index.bin')  # 将索引保存到文件中，方便以后使用
    return index

def find_k_similar(xq, index, k=50):
    # 设置ef（搜索时的搜索质量参数），并在索引上进行批量查询
    query_params = {'ef': 4000, 'searchMethod': 3}  # 设置搜索时的参数，ef是搜索质量参数，越大越准确，但是速度越慢，searchMethod是查询算法的选择，3是最快的，也是最推荐的
    index.setQueryTimeParams(query_params)  # 设置查询时的参数
    start_time = time.time()
    # 实际查询
    I = index.knnQueryBatch(xq, k=k, num_threads=4)
    # 记录查询结束时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Query Time: {total_time:.4f} seconds")
    ans = [x[0] for x in I]
    indices = []
    for i in range(len(I)):
        # 取出lst[i][0]中的50个数据，作为一行
        row = I[i][0]
        # 在row的开头插入索引i，用numpy的concatenate方法
        #  row = np.concatenate(([i], row))
        # 把row添加到indices中
        indices.append(row)
    return indices


# 定义向量的维度和查询的数量
d = 512 # 向量维度
nq = 500 # 查询数量
k = 50  # 每个查询向量要返回的最近邻数量

xb = np.load('gallery_emb.npy')
xq = np.load('query_emb.npy')

index = build_index(xb)
indices = find_k_similar(xq, index, k)

with open('labels_500.pkl', 'rb') as f:
    true_labels = pickle.load(f)

correct_count = 0
for query_idx, true_neighbors in true_labels.items():
    correct_neighbors = set(true_neighbors) & set(indices[query_idx])
    correct_count += len(correct_neighbors)

accuracy = correct_count / (len(true_labels) * k)
print(f"Accuracy: {accuracy * 100:.2f}%")




