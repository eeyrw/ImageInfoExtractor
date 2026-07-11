import psycopg2
import numpy as np
import faiss

BATCH_SIZE = 5000  # 每次批量插入数据库的数量

# ---------- 数据库连接 ----------
conn = psycopg2.connect(
    dbname="image_dataset_db4",
    user="postgres",
    password="example",
    host="localhost",
    port=5432
)
cur = conn.cursor(name="pose_cursor")
cur.itersize = 100000

# ---------- 读取每张图片 bbox 最大的 pose ----------
cur.execute("""
SELECT image_id, pose_index, kpts_x, kpts_y, invalid_kpts_idx, bbox
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY image_id ORDER BY bbox[3]*bbox[4] DESC) AS rn
    FROM image_pose
    WHERE kpts_x IS NOT NULL AND kpts_y IS NOT NULL AND bbox IS NOT NULL
) t
WHERE rn = 1;
""")

pose_ids = []
pose_indices = []
pose_vectors = []
pose_bboxes = []

for row in cur:
    img_id, pose_index, kpts_x, kpts_y, invalid_kpts_idx, bbox = row
    vec = np.array(
        [0.0 if idx in invalid_kpts_idx else kpts_x[idx] for idx in range(len(kpts_x))] +
        [0.0 if idx in invalid_kpts_idx else kpts_y[idx] for idx in range(len(kpts_y))],
        dtype=np.float32
    )

    pose_ids.append(img_id)
    pose_indices.append(pose_index)
    pose_vectors.append(vec)
    pose_bboxes.append(bbox)

cur.close()
conn.close()

X = np.array(pose_vectors, dtype=np.float32)
print("Shape:", X.shape)

# ---------- Faiss 聚类 ----------
d = X.shape[1]
n_clusters = 1000
kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True, gpu=True)
kmeans.train(X)
_, labels = kmeans.index.search(X, 1)
labels = labels.reshape(-1)

# ---------- 批量插入 pose_clusters 表 ----------
conn = psycopg2.connect(
    dbname="image_dataset_db4",
    user="postgres",
    password="example",
    host="localhost",
    port=5432
)
cur = conn.cursor()

for start in range(0, len(pose_ids), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch_data = [
        (pose_ids[i], pose_indices[i], int(labels[i]), pose_bboxes[i], pose_vectors[i].tolist())
        for i in range(start, min(end, len(pose_ids)))
    ]

    args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s)", x).decode('utf-8') for x in batch_data)
    cur.execute(f"INSERT INTO pose_clusters (image_id, pose_index, cluster_id, bbox, embedding) VALUES {args_str} ON CONFLICT (image_id, pose_index) DO UPDATE SET cluster_id=EXCLUDED.cluster_id, bbox=EXCLUDED.bbox, embedding=EXCLUDED.embedding;")

conn.commit()
cur.close()
conn.close()

print("✅ Pose 聚类结果已批量写入 pose_clusters 表")
