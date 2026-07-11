import psycopg2
import numpy as np
import faiss

from ClusteringTree import print_cluster_sizes, recursive_kmeans_gpu_elbow  # pip install faiss-cpu 或 faiss-gpu

# ---------- 数据库连接 ----------
conn = psycopg2.connect(
    dbname="image_dataset_db4",
    user="postgres",
    password="example",
    host="localhost",
    port=5432
)
cur = conn.cursor(name="embedding_cursor")

# ---------- 流式取数据 ----------
cur.itersize = 100000
cur.execute("SELECT id, image_embedding FROM images WHERE image_embedding IS NOT NULL;")

ids = []
embeddings = []

for row in cur:
    ids.append(row[0])
    embeddings.append(row[1])  # list[float]

cur.close()
conn.close()

# ---------- 转 numpy ----------
X = np.array(embeddings, dtype=np.float32)
print("Shape:", X.shape)  # (N, 1024)

labels_list = recursive_kmeans_gpu_elbow(
    X,
    max_size=1000,
    gpu_id=0,
    kmeans_niter=5,
    max_k=2000,
    num_candidates=20,
    min_points_per_cluster=50,
    verbose=True
)

print("\n每层簇大小统计:")
print_cluster_sizes(labels_list)

# ---------- Faiss KMeans ----------
# d = X.shape[1]  # 维度 = 1024
# n_clusters = 5000

# kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True, gpu=True)  # gpu=True 用 GPU
# kmeans.train(X)

# 获取聚类标签
# _, labels = kmeans.index.search(X, 1)
# labels = labels.reshape(-1)

# # ---------- 写回数据库 ----------
# conn = psycopg2.connect(
#     dbname="your_db",
#     user="your_user",
#     password="your_pass",
#     host="localhost",
#     port=5432
# )
# cur = conn.cursor()

# cur.execute("ALTER TABLE images ADD COLUMN IF NOT EXISTS cluster_id INTEGER;")

# for img_id, label in zip(ids, labels):
#     cur.execute("UPDATE images SET cluster_id = %s WHERE id = %s", (int(label), img_id))

# conn.commit()
# cur.close()
# conn.close()

print("✅ Faiss 聚类完成，结果已写回 cluster_id 列")
