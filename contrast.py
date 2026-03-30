import numpy as np
import pyo3_lib
import ctypes
import os

# 1. 创建两个测试矩阵
A = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
B = np.array([[2.0, 1.0], [-1.0, 3.0]], dtype=np.float64)

print("Starting Matmul with ReLU...")

# ==========================================
# [路线一] PyO3 调用
# ==========================================
C_pyo3 = pyo3_lib.matmul_relu_pyo3(A, B)
print("Pyo3 version Result:\n", C_pyo3)


# ==========================================
# [路线二] ctypes 调用
# ==========================================
lib_path = os.path.join(os.path.dirname(__file__), "rs_kernel", "debug", "release", "librs_kernel.so")
rs_lib = ctypes.cdll.LoadLibrary(lib_path)

# 指定 C 接口签名
rs_lib.matmul_relu_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t
]
rs_lib.matmul_relu_c.restype = None

# 1. 动态提取矩阵维度
M, K = A.shape
K2, N = B.shape
assert K == K2, "矩阵维度不匹配" 

# 2. 内存布局对齐 (非常关键！)
# 确保 numpy 数组在内存中是 C 连续的 (C-contiguous)
A_c = np.ascontiguousarray(A, dtype=np.float64)
B_c = np.ascontiguousarray(B, dtype=np.float64)
# 提前在 Python 端分配好输出矩阵 C 的内存空间，全部填 0
C_ctypes = np.zeros((M, N), dtype=np.float64)

# 3. 提取裸指针 (零拷贝魔法：直接把 numpy 底层 C 数组的指针传给 Rust)
a_ptr = A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
b_ptr = B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
c_ptr = C_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

print("[*] 开始执行 Rust (ctypes) 版本的算子...")

# 4. 跨界调用！
rs_lib.matmul_relu_c(a_ptr, b_ptr, c_ptr, M, K, N)

print("ctypes version Result:\n", C_ctypes)

# 验证两种方式结果是否一致
np.testing.assert_allclose(C_pyo3, C_ctypes, err_msg="两边计算结果不一致！")
print("✅ PyO3 和 ctypes 结果完全匹配！")
