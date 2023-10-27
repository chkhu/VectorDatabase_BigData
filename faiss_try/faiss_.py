# 10.28 01:52 test_a:
# Total Query Time: 1.03 seconds
# Average Query Time per Vector: 2.06 ms
# Accuracy: 99.80%

import numpy as np
import faiss
import pickle
import time

# 查询最近的 K 个向量
K = 10

# 加载查询向量和大规模向量库
query_emb = np.load("test_a/query_emb.npy").astype('float32')
G = np.load("test_a/gallery_emb.npy").astype('float32')

# 设定向量的维度
d = query_emb.shape[1]

# 建立 Faiss 索引
index = faiss.IndexFlatL2(d)
index.add(G)

# 开始计时
start_time = time.time()

# 查询
_, faiss_indices = index.search(query_emb, K)

# 结束计时
end_time = time.time()

# 计算查询时间
total_query_time = end_time - start_time
average_query_time = total_query_time / len(query_emb)

print(f"Total Query Time: {total_query_time:.2f} seconds")
print(f"Average Query Time per Vector: {average_query_time * 1000:.2f} ms")

# 加载真实的查询结果（从labels_500.pkl文件中）
with open('test_a/labels_500.pkl', 'rb') as f:
    true_labels = pickle.load(f)

# 评估性能
correct_count = 0
for query_idx, true_neighbors in true_labels.items():
    correct_neighbors = set(true_neighbors) & set(faiss_indices[query_idx])
    correct_count += len(correct_neighbors)

accuracy = correct_count / (len(true_labels) * K)
print(f"Accuracy: {accuracy * 100:.2f}%")