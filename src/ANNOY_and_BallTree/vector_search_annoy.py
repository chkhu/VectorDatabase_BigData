import time
import pickle
import numpy as np
from annoy import AnnoyIndex

# 读取保存向量的NumPy文件
vectors = np.load('gallery_emb.npy')

# 定义向量的维度
vector_dimension = vectors.shape[1]

# 定义Annoy的索引文件，指定向量的维度
annoy_index = AnnoyIndex(vector_dimension)

# 添加向量到Annoy的索引中
for i in range(vectors.shape[0]):
    annoy_index.add_item(i, vectors[i])

# 构建Annoy的向量投影树，指定树的数量（可以根据需求调整）
num_trees = 500
annoy_index.build(num_trees)

# 保存Annoy的索引到文件，以便后续使用
# annoy_index.save('annoy_index.ann')

# 在后续的查询中，你可以加载这个索引文件，使用Annoy进行最近邻搜索
# annoy_index = AnnoyIndex(vector_dimension)
# annoy_index.load('annoy_index.ann')

# 读取保存查询向量的NumPy文件
query_vectors = np.load('query_emb.npy')

# 查询每个查询向量的最近邻
num_neighbors = 10  # 你希望找到的最近邻的数量

# 加载真实的查询结果（从labels_500.pkl文件中）
with open('labels_500.pkl', 'rb') as f:
    true_labels = pickle.load(f)

# 计算Annoy的平均正确率和平均时间
total_correct = 0
total_time = 0
total_queries = len(true_labels)

# 对每个查询进行Annoy检索并比较结果
for query_idx, true_neighbors in true_labels.items():
    query_vector = query_vectors[query_idx]
    # 记录查询开始时间
    start_time = time.time()
    # 实际查询
    annoy_neighbors = annoy_index.get_nns_by_vector(query_vector, num_neighbors)
    # 记录查询结束时间
    end_time = time.time()
    total_time += (end_time - start_time)
    correct_neighbors = set(true_neighbors) & set(annoy_neighbors)
    print(correct_neighbors)
    accuracy = len(correct_neighbors) / num_neighbors
    total_correct += accuracy

average_accuracy = total_correct / total_queries
average_time = total_time / total_queries

print("Annoy 的平均正确率为: {:.2f}%".format(average_accuracy * 100))
print("Annoy 的平均查询用时为: {:.3f}s".format(average_time))
