import timeit
import numpy as np
import ctypes
import os
import pyo3_lib

# ==========================================
# 1. 准备 ctypes 环境 (提前加载，不计入耗时)
# ==========================================
lib_path = os.path.join(os.path.dirname(__file__), "rs_kernel", "target", "release", "librs_kernel.so")
rs_lib = ctypes.cdll.LoadLibrary(lib_path)

rs_lib.matmul_relu_c.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
]
rs_lib.matmul_relu_c.restype = None

# ==========================================
# 2. 压测核心逻辑
# ==========================================
def benchmark(size, iterations):
    print(f"\n[*] 开始压测: 矩阵维度 {size}x{size}, 循环 {iterations} >次")

    # 初始化连续内存矩阵
    A = np.random.rand(size, size).astype(np.float64)
    B = np.random.rand(size, size).astype(np.float64)
    C_ctypes = np.zeros((size, size), dtype=np.float64)

    # ctypes 指针提取
    a_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    b_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_ptr = C_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # --- 测试 PyO3 ---
    pyo3_time = timeit.timeit(lambda: pyo3_lib.matmul_relu_pyo3(A, B), number=iterations)
    print(f"  -> PyO3 耗时:   {pyo3_time:.5f} 秒")

    # --- 测试 ctypes ---
    ctypes_time = timeit.timeit(lambda: rs_lib.matmul_relu_c(a_ptr, b_ptr, c_ptr, size, size, size), number=iterations)
    print(f"  -> ctypes 耗时: {ctypes_time:.5f} 秒")


# ==========================================
# 3. 运行对比
# ==========================================
if __name__ == "__main__":
    # 场景一：微小矩阵 (2x2)，放大 FFI 纯胶水开销
    #benchmark(size=2, iterations=5_000_000)

    # 场景二：中型矩阵 (200x200)，计算量 O(N^3) 开始占据主导i
    benchmark(size=200, iterations=500)
