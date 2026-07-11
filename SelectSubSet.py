import psycopg2
import numpy as np
import faiss
import polars as pl
import uuid

# ========== 参数 ==========
N_CLUSTER = 50000
N_SELECT = 10000
PARQUET_FILE = "representative_sample.parquet"

# ========== 数据库连接 (流式读取) ==========
conn = psycopg2.connect(
    dbname="image_dataset_db4",
    user="postgres",
    password="example",
    host="localhost",
    port=5432
)
cur = conn.cursor(name="embedding_cursor")
cur.itersize = 100000

cur.execute("SELECT id, image_embedding FROM images WHERE image_embedding IS NOT NULL;")

ids = []
embeddings = []
for row in cur:
    ids.append(row[0])          # UUID
    embeddings.append(row[1])   # list[float]

cur.close()
conn.close()

X = np.array(embeddings, dtype=np.float32)
print("Shape:", X.shape)  # (N, 1024)
d = X.shape[1]

# ========== Step 1: GPU KMeans ==========
res = faiss.StandardGpuResources()
gpu_index = faiss.GpuIndexFlatL2(res, d)
gpu_index.add(X)

kmeans = faiss.Clustering(d, N_CLUSTER)
kmeans.niter = 25
kmeans.verbose = True
kmeans.train(X, gpu_index)

centroids = faiss.vector_to_array(kmeans.centroids).reshape(N_CLUSTER, d)

# ========== Step 2: K-Center Greedy ==========
def kcenter_greedy(X, K, seed=123):
    np.random.seed(seed)
    N, D = X.shape
    first = np.random.randint(0, N)
    selected = [first]
    min_dists = np.sum((X - X[first])**2, axis=1)
    for it in range(1, K):
        idx = np.argmax(min_dists)
        selected.append(idx)
        d2 = np.sum((X - X[idx])**2, axis=1)
        min_dists = np.minimum(min_dists, d2)
    return np.array(selected, dtype=np.int64)

sel_centroid_idx = kcenter_greedy(centroids, N_SELECT)
selected_centroids = centroids[sel_centroid_idx]

# ========== Step 3: 找回原始样本 ==========
index_flat = faiss.GpuIndexFlatL2(res, d)
index_flat.add(X)

D, I = index_flat.search(selected_centroids, 1)
selected_indices = I.flatten()
selected_image_ids = [ids[i] for i in selected_indices]  # UUID list

# ========== Step 4: 存到 favorites ==========
conn = psycopg2.connect(
    dbname="image_dataset_db4",
    user="postgres",
    password="example",
    host="localhost",
    port=5432
)
cur = conn.cursor()

# 创建收藏集
favorite_id = str(uuid.uuid4())
cur.execute(
    """
    INSERT INTO favorites (id, name, path, description)
    VALUES (%s, %s, %s, %s)
    """,
    (favorite_id, "Representative Sample", "/auto/representative_sample", "自动抽样代表集")
)

# 批量插入 favorite_images
args = [(favorite_id, img_id) for img_id in selected_image_ids]
cur.executemany(
    "INSERT INTO favorite_images (favorites_id, image_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
    args
)

conn.commit()
cur.close()
conn.close()

print(f"✅ 已存入 favorites({favorite_id})，包含 {len(selected_image_ids)} 张图片")

# ========== Step 5: 导出到 Parquet ==========
df = pl.DataFrame({
    "image_id": selected_image_ids,
    "favorite_id": [favorite_id] * len(selected_image_ids)
})
df.write_parquet(PARQUET_FILE)
print(f"✅ 已导出 {PARQUET_FILE}")
