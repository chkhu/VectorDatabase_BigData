1. 确保有可用的python版本，并安装以下依赖

   ```
   pip install numpy
   pip install faiss-cpu  # CPU版本
   # 或者
   pip install faiss-gpu  # GPU版本（如果有NVIDIA显卡和CUDA）
   ```

2. 将test_a数据集文件夹移动到根目录。或者，使用代码编辑器把代码中下面的

   ```
   test_a/query_emb.npy
   ...
   test_a/gallery_emb.npy
   ...
   test_a/labels_500.pkl
   ```

   改成你存放数据集文件的路径。

3. 运行

   ```
   python faiss_FLAT.py
   ```