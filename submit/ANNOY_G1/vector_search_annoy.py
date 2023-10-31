# 10.31 16:44 test_a:
# Average Query Time per Vector: 1 ms
# Accuracy: 88.72%

import time
import pickle
import numpy as np
from annoy import AnnoyIndex

# 读取保存向量的NumPy文件
data_vectors = np.load('gallery_emb.npy')

# 读取保存查询向量的NumPy文件
query_vectors = np.load('query_emb.npy')

# 建立 Annoy 索引
def build_index(data_vectors):

    # 定义向量的维度
    vector_dimension = data_vectors.shape[1]

    # 定义Annoy的索引文件，指定向量的维度
    annoy_index = AnnoyIndex(vector_dimension)

    # 添加向量到Annoy的索引中
    for i in range(data_vectors.shape[0]):
        annoy_index.add_item(i, data_vectors[i])

    # 构建Annoy的向量投影树，指定树的数量（可以根据需求调整）
    num_trees = 500
    annoy_index.build(num_trees)

    # 保存Annoy的索引到文件，以便后续使用
    # annoy_index.save('annoy_index.ann')

    # 在后续的查询中，你可以加载这个索引文件，使用Annoy进行最近邻搜索
    # annoy_index = AnnoyIndex(vector_dimension)
    # annoy_index.load('annoy_index.ann')

    return annoy_index

def find_k_similar(query_vectors, index, k=50):

    # 加载真实的查询结果（从labels_500.pkl文件中）
    # with open('labels_500.pkl', 'rb') as f:
    #     true_labels = pickle.load(f)

    # 计算Annoy的平均正确率和平均时间
    # total_correct = 0
    total_time = 0
    # total_queries = len(true_labels)

    indices = []

    # 对每个查询进行Annoy检索并比较结果
    for query_idx in range(query_vectors.shape[0]):
        query_vector = query_vectors[query_idx]
        # 记录查询开始时间
        start_time = time.time()
        # 实际查询
        annoy_neighbors = index.get_nns_by_vector(query_vector, k)
        # 记录查询结束时间
        end_time = time.time()
        total_time += (end_time - start_time)
        # correct_neighbors = set(true_neighbors) & set(annoy_neighbors)
        indices.append(list(annoy_neighbors))
        # print(correct_neighbors)
        # accuracy = len(correct_neighbors) / k
        # total_correct += accuracy

    # average_accuracy = total_correct / total_queries
    # average_time = total_time / total_queries

    # print("Annoy 的平均正确率为: {:.2f}%".format(average_accuracy * 100))
    # print("Annoy 的平均查询用时为: {:.3f}s".format(average_time))

    return indices

index = build_index(data_vectors)
print(find_k_similar(query_vectors, index, 10))
