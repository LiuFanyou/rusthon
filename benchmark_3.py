import timeit
import numpy as np
import ctypes
import os

import pyo3_lib  

# ==========================================
# 1. 准备 ctypes 环境
# ==========================================
# 假设你的 rs_kernel 依然独立存在，这里直接加载
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
    print(f"\n[*] 开始压测: 矩阵维度 {size}x{size}, 循环 {iterations} 次")

    # 初始化连续内存矩阵
    A = np.random.rand(size, size).astype(np.float64)
    B = np.random.rand(size, size).astype(np.float64)
    C_out = np.zeros((size, size), dtype=np.float64) # 用于原地修改的输出矩阵

    # --- 提前提取指针与物理地址 (不计入耗时) ---
    # ctypes 用的 ctypes.POINTER 对象
    a_ptr_ctypes = A.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    b_ptr_ctypes = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_ptr_ctypes = C_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # PyO3 缝合怪用的 usize 物理地址 (整数)
    a_addr = A.ctypes.data
    b_addr = B.ctypes.data
    c_addr = C_out.ctypes.data

    # ================= 对决开始 =================

    # 1. 测试 安全的 PyO3 (买保险、戴安全帽版)
   
   #pyo3_safe_time = timeit.timeit(
    #    lambda: pyo3_lib.matmul_relu_pyo3(A, B), 
     #   number=iterations
    #)
#    print(f"  -> PyO3 (安全版) 耗时:   {pyo3_safe_time:.5f} 秒")
    
    # 2. 测试 ctypes (动态安检、垃圾回收版)
    ctypes_time = timeit.timeit(
        lambda: rs_lib.matmul_relu_c(a_ptr_ctypes, b_ptr_ctypes, c_ptr_ctypes, size, size, size), 
        number=iterations
    )
    print(f"  -> ctypes (动态版) 耗时: {ctypes_time:.5f} 秒")

    # 3. 测试 极限缝合怪 PyO3 (Vectorcall + 裸指针)
    frankenstein_time = timeit.timeit(
        lambda: pyo3_lib.matmul_relu_frankenstein(a_addr, b_addr, c_addr, size, size, size), 
        number=iterations
    )
    print(f"  -> PyO3 (缝合怪) 耗时:   {frankenstein_time:.5f} 秒")

    # ================= 数据统计 =================
    if ctypes_time > 0:
        print(f"\n  => 倍率参考:")
        print(f"     [降维打击] 缝合怪     是 ctypes 的 {frankenstein_time / ctypes_time:.2f} 倍开销")

# ==========================================
# 3. 运行对比
# ==========================================
if __name__ == "__main__":
    # 场景一：极限榨取胶水开销
    benchmark(size=2, iterations=1_000_000)

    # 场景二：中型计算
    # benchmark(size=200, iterations=500)
