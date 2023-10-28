# 10.28 02:22 test_a:
#Total Query Time: 0.06 seconds
# Average Query Time per Vector: 0.12 ms
# Accuracy: 55.08%

import numpy as np
import faiss
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

# 查询最近的 K 个向量
K = 10

# 加载查询向量和大规模向量库
query_emb = np.load("test_a/query_emb.npy").astype('float32')
G = np.load("test_a/gallery_emb.npy").astype('float32')

# 对数据进行归一化，以便使用余弦相似度
faiss.normalize_L2(query_emb)
faiss.normalize_L2(G)

# 设定向量的维度
d = query_emb.shape[1]

# 建立 Faiss 索引
quantizer = faiss.IndexFlatL2(d)  # The quantizer
index = faiss.IndexIVFFlat(quantizer, d, int(np.sqrt(G.shape[0])))  # Adjusting clusters to sqrt(N)
index.train(G)
index.add(G)

# 将查询分解成多个子查询，以便并发执行
def chunked_query(query_chunk):
    _, indices_chunk = index.search(query_chunk, K)
    # 保存结果到result.pkl
    with open('result.pkl', 'wb') as f:
        pickle.dump(indices_chunk, f)
    return indices_chunk

# 使用并发进行查询并计时
chunks = np.array_split(query_emb, 10)  # 10 can be adjusted depending on the number of CPU cores

start_time = time.time()
with ThreadPoolExecutor() as executor:
    results = list(executor.map(chunked_query, chunks))
end_time = time.time()

# 将结果合并
faiss_indices = np.vstack(results)

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
