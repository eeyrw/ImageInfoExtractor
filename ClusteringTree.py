import faiss
import numpy as np
from typing import List, Tuple

def recursive_kmeans_gpu_elbow(
    vectors: np.ndarray,
    max_size: int,
    gpu_id: int = 0,
    kmeans_niter: int = 20,
    max_k: int = 250,
    num_candidates: int = 5,
    min_points_per_cluster: int = 50,
    verbose: bool = False
) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    GPU + 递归 KMeans + Elbow 方法自动选择簇数
    
    参数:
        vectors: np.ndarray, shape (N, d), dtype=float32
        max_size: 当子簇大小 <= max_size 时停止递归
        gpu_id: 使用的 GPU 编号
        kmeans_niter: KMeans 迭代次数
        max_k: 最大簇数限制
        num_candidates: 候选簇数数量
        min_points_per_cluster: 每簇最少点数
        verbose: 是否打印调试信息
    
    返回:
        labels_list: List[List[Tuple[np.ndarray, np.ndarray]]]
        每层存 [(vecs_idx, labels_local)]，vecs_idx 为原始向量索引
    """
    assert vectors.dtype == np.float32, "vectors 必须为 float32"
    N, d = vectors.shape
    labels_list: List[List[Tuple[np.ndarray, np.ndarray]]] = []

    # 创建 GPU 资源
    gpu_res = faiss.StandardGpuResources()

    # -----------------------------
    # 生成候选簇数序列
    # -----------------------------
    def generate_candidate_ks(max_k: int, num_candidates: int):
        # 使用指数增长序列，最小簇数 2
        seq = np.logspace(np.log10(2), np.log10(max_k), num=num_candidates)
        seq = np.unique(np.round(seq).astype(int))
        return seq.tolist()

    # -----------------------------
    # 递归聚类函数
    # -----------------------------
    def _recursive_cluster(vecs_idx: np.ndarray, level: int):
        n_vecs = len(vecs_idx)
        # 递归终止条件：空簇或子簇大小 <= max_size
        if n_vecs == 0 or n_vecs <= max_size:
            if verbose:
                print(f"Level {level}: stop splitting (n_vecs={n_vecs} <= max_size={max_size})")
            return

        sub_vectors = vectors[vecs_idx]

        # -----------------------------
        # 生成候选簇数，并约束 min_points_per_cluster
        # -----------------------------
        candidate_ks = generate_candidate_ks(max_k, num_candidates)
        max_possible_k = max(1, n_vecs // min_points_per_cluster)
        candidate_ks = [k for k in candidate_ks if k <= max_possible_k]
        if len(candidate_ks) == 0:
            candidate_ks = [1]  # 防止空序列

        sse_list = []

        # -----------------------------
        # 训练每个候选簇数 KMeans 并计算 SSE (GPU search 优化)
        # -----------------------------
        for k in candidate_ks:
            # GPU KMeans 训练
            clustering = faiss.Clustering(d, k)
            clustering.niter = kmeans_niter
            clustering.verbose = verbose

            # 构建 GPU IndexFlatL2
            index_flat = faiss.IndexFlatL2(d)
            index_gpu = faiss.index_cpu_to_gpu(gpu_res, gpu_id, index_flat)
            clustering.train(sub_vectors, index_gpu)

            # 获取质心
            centroids = faiss.vector_to_array(clustering.centroids).reshape(k, d).astype(np.float32)

            # 使用 FAISS GPU search 计算 SSE
            centroid_index_cpu = faiss.IndexFlatL2(d)
            centroid_index_cpu.add(centroids)
            centroid_index_gpu = faiss.index_cpu_to_gpu(gpu_res, gpu_id, centroid_index_cpu)
            D, _ = centroid_index_gpu.search(sub_vectors, 1)
            sse = np.sum(D)  # D 已经是平方距离

            sse_list.append(sse)

            # 清理 GPU 索引
            del index_gpu, centroid_index_gpu, centroid_index_cpu

        # -----------------------------
        # Elbow 方法选择簇数
        # -----------------------------
        sse_array = np.array(sse_list)
        if len(sse_array) >= 3:
            second_diff = np.diff(sse_array, 2)       # 二阶差分
            elbow_idx = np.argmin(second_diff) + 1    # 拐点索引 +1 对应原候选簇数
        else:
            elbow_idx = 0                             # 候选簇数太少时直接取第一个
        best_k = candidate_ks[elbow_idx]

        if verbose:
            print(f"Level {level}: n_vecs={n_vecs}, candidate_ks={candidate_ks}, SSE={sse_list}, selected k={best_k}")

        # -----------------------------
        # 最终 GPU KMeans
        # -----------------------------
        clustering = faiss.Clustering(d, best_k)
        clustering.niter = kmeans_niter
        clustering.verbose = verbose
        index_flat = faiss.IndexFlatL2(d)
        index_gpu = faiss.index_cpu_to_gpu(gpu_res, gpu_id, index_flat)
        clustering.train(sub_vectors, index_gpu)

        # 获取最终质心
        centroids = faiss.vector_to_array(clustering.centroids).reshape(best_k, d).astype(np.float32)

        # -----------------------------
        # GPU search 计算最终 labels
        # -----------------------------
        centroid_index_cpu = faiss.IndexFlatL2(d)
        centroid_index_cpu.add(centroids)
        centroid_index_gpu = faiss.index_cpu_to_gpu(gpu_res, gpu_id, centroid_index_cpu)
        D, labels_local = centroid_index_gpu.search(sub_vectors, 1)
        labels_local = labels_local.ravel().astype(np.int32)

        # -----------------------------
        # 保存当前层标签
        # -----------------------------
        while len(labels_list) <= level:
            labels_list.append([])
        labels_list[level].append((vecs_idx.copy(), labels_local.copy()))

        # -----------------------------
        # 递归处理子簇
        # -----------------------------
        for k in range(best_k):
            mask = (labels_local == k)
            if not np.any(mask):
                continue
            child_vecs_idx = vecs_idx[mask]
            _recursive_cluster(child_vecs_idx, level + 1)

        del index_gpu, centroid_index_gpu  # 清理 GPU 资源

    # -----------------------------
    # 开始递归
    # -----------------------------
    all_idx = np.arange(N, dtype=np.int32)
    _recursive_cluster(all_idx, 0)
    return labels_list


# -----------------------------
# 打印每层簇大小
# -----------------------------
def print_cluster_sizes(labels_list: List[List[Tuple[np.ndarray, np.ndarray]]]):
    for level, layer in enumerate(labels_list):
        print(f"Level {level}:")
        for i, (vecs_idx, labels) in enumerate(layer):
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"  Sub-cluster {i}-{u}: {c} points")
        print("-" * 40)




# -----------------------------
# 测试示例
# -----------------------------
if __name__ == "__main__":
    N, d = 500000, 1024
    rng = np.random.RandomState(123)
    xb = rng.random_sample((N, d)).astype('float32')

    labels_list = recursive_kmeans_gpu_elbow(
        xb,
        max_size=1000,
        gpu_id=0,
        kmeans_niter=5,
        max_k=2000,
        num_candidates=20,
        min_points_per_cluster=10,
        verbose=True
    )

    print("\n每层簇大小统计:")
    print_cluster_sizes(labels_list)