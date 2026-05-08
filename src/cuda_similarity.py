import os
import ctypes
import subprocess
import numpy as np
from time import perf_counter as pc

CUDA_SRC = "cosine.cu"
SO_FILE = "libcosine.so"


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


lib.destroy_index.argtypes = [ctypes.c_void_p]

lib.destroy_index.restype = None


class CosineIndex:

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

    def __del__(self):

        if hasattr(self, "index_ptr"):
            lib.destroy_index(self.index_ptr)


if __name__ == "__main__":

    # compare the retrieval time with numpy
    M = 100000
    N = 1024
    matrix = np.random.randn(M, N).astype(np.float32)

    index = CosineIndex(matrix)
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
