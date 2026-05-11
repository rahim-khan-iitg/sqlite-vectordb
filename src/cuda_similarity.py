import os
import ctypes
import subprocess
import numpy as np
from src.models import DistanceMetric
from time import perf_counter as pc

CUDA_SRC = "distance_kernels.cu"
SO_FILE = "libdistance.so"


if not os.path.exists(SO_FILE):

    compile_cmd = [
        "nvcc",
        "-arch=sm_86",
        "-O3",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        CUDA_SRC,
        "-o",
        SO_FILE,
    ]

    print("Compiling CUDA code...")
    subprocess.check_call(compile_cmd)
    print("Compilation done.")


lib = ctypes.cdll.LoadLibrary(f"./{SO_FILE}")


lib.create_index.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]

lib.create_index.restype = ctypes.c_void_p


lib.search_index.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
]

lib.search_index.restype = None


lib.knn_search.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_char_p,
]

lib.knn_search.restype = None


lib.destroy_index.argtypes = [ctypes.c_void_p]

lib.destroy_index.restype = None


class Index:

    def __init__(self, matrix):

        matrix = np.asarray(matrix, dtype=np.float32)

        assert matrix.ndim == 2

        self.matrix = matrix
        self.M, self.N = matrix.shape

        self.index_ptr = lib.create_index(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.M, self.N
        )

    def search(self, query):

        query = np.asarray(query, dtype=np.float32)

        assert query.ndim == 1
        assert query.shape[0] == self.N

        scores = np.zeros(self.M, dtype=np.float32)

        lib.search_index(
            self.index_ptr,
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.M,
            self.N,
        )

        return scores

    def search_knn(self, query, k, metric: DistanceMetric):

        query = np.asarray(query, dtype=np.float32)

        assert query.ndim == 1
        assert query.shape[0] == self.N

        scores = np.zeros(self.M, dtype=np.float32)

        lib.knn_search(
            self.index_ptr,
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.M,
            self.N,
            metric.value.encode("utf-8"),
        )

        # Get indices of k smallest distances
        k = min(k, self.M)
        indices = np.argpartition(scores, k)[:k]
        indices = indices[np.argsort(scores[indices])]

        return indices, scores[indices]

    def __del__(self):

        if hasattr(self, "index_ptr"):
            lib.destroy_index(self.index_ptr)


if __name__ == "__main__":

    # compare the retrieval time with numpy
    M = 10000
    N = 1024
    matrix = np.random.randn(M, N).astype(np.float32)

    index = Index(matrix)
    numpy_time = 0.0
    cuda_time = 0.0
    for i in range(5):

        query = np.random.randn(N).astype(np.float32)

        # numpy time
        start = pc()
        scores_np = np.dot(matrix, query)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        scores_np = scores_np / (norms + 1e-8)
        end = pc()
        numpy_time += end - start
        # cuda time
        start = pc()
        scores_cuda = index.search(query)
        end = pc()
        print(np.linalg.norm(scores_np - scores_cuda, ord=2))
        cuda_time += end - start
    print(f"Numpy time: {numpy_time/5*1000} ms")
    print(f"Cuda time: {cuda_time/5*1000} ms")
    print("\nTesting search_knn for different metrics:")
    k = 10
    for metric in DistanceMetric:
        print(f"\nMetric: {metric.name}")
        query = np.random.randn(N).astype(np.float32)

        # CUDA result
        start = pc()
        cuda_indices, cuda_distances = index.search_knn(query, k, metric)
        cuda_time = (pc() - start) * 1000

        # Ground truth (NumPy)
        start = pc()
        if metric == DistanceMetric.COSINE:
            # Distance = 1 - Cosine Similarity
            dot = np.dot(matrix, query)
            norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
            gt_scores = 1.0 - (dot / (norms + 1e-8))
        elif metric == DistanceMetric.EUCLIDEAN:
            gt_scores = np.linalg.norm(matrix - query, axis=1)
        elif metric == DistanceMetric.MANHATTAN:
            gt_scores = np.sum(np.abs(matrix - query), axis=1)

        gt_indices = np.argsort(gt_scores)[:k]
        gt_distances = gt_scores[gt_indices]
        numpy_time = (pc() - start) * 1000

        print(f"  Cuda time: {cuda_time:.4f} ms")
        print(f"  Numpy time: {numpy_time:.4f} ms")

        # Verify indices match (handling potential ties in distances is rare with random data)
        # We check if the set of indices matches if distances are very close,
        # but usually direct comparison is fine.
        idx_diff = np.setdiff1d(cuda_indices, gt_indices)
        if len(idx_diff) == 0:
            print("  ✅ Indices match!")
        else:
            # If indices don't match exactly, check if distances are the same (ties)
            if np.allclose(gt_scores[cuda_indices], gt_scores[gt_indices]):
                print("  ✅ Indices differ but distances are identical (tie case)!")
            else:
                print(
                    f"  ❌ Indices mismatch! CUDA: {cuda_indices[:5]}... vs GT: {gt_indices[:5]}..."
                )

        # Verify distances match
        dist_err = np.linalg.norm(cuda_distances - gt_distances)
        if dist_err < 1e-5:
            print(f"  ✅ Distances match! (Error: {dist_err:.2e})")
        else:
            print(f"  ❌ Distance mismatch! Error: {dist_err:.2e}")
