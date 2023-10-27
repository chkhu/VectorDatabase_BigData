import time
import pickle
import numpy as np
from sklearn.neighbors import BallTree

# 读取保存向量的NumPy文件
gallery_vectors = np.load('gallery_emb.npy')
query_vectors = np.load('query_emb.npy')

# 构建索引
num_neighbors = 10  # 每个查询的最近邻数量
# 第二个参数是 leaf_size，也就是树的 n 叉设置，可以一定程度调整速度，但效果有限
ball_tree = BallTree(gallery_vectors, 60)

# 加载真实的查询结果（从labels_500.pkl文件中）
with open('labels_500.pkl', 'rb') as f:
    true_labels = pickle.load(f)

# 存储查询时间的列表
query_times = []
total_correct = 0

# 对每个查询进行检索并比较结果
for query_idx, true_neighbors in true_labels.items():
    query_vector = query_vectors[query_idx].reshape(1, -1)  # 将查询向量转换为二维数组
    
    # 记录查询开始时间
    start_time = time.time()
    
    # 在BallTree索引中查询最近邻
    distances, indices = ball_tree.query(query_vector, k=num_neighbors)
    ball_tree_neighbors = indices[0]
    
    # 记录查询结束时间
    end_time = time.time()
    
    # 计算查询时间
    query_time = end_time - start_time
    query_times.append(query_time)
    
    # 比较结果等部分略去，ball_tree_neighbors中包含了 BallTree 的查询结果
    correct_neighbors = set(true_neighbors) & set(ball_tree_neighbors)
    print(correct_neighbors)
    accuracy = len(correct_neighbors) / num_neighbors
    total_correct += accuracy

# 计算平均查询时间和正确率
average_accuracy = total_correct / len(query_times)
average_query_time = sum(query_times) / len(query_times)

print("BallTree 的平均正确率为: {:.2f}%".format(average_accuracy * 100))
print("BallTree 的平均查询用时为: {:.3f}s".format(average_query_time))
