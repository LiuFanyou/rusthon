import time
import random

# ==========================================
# 模块一：纯 Python 基准 (对照组)
# 模拟神经网络中的： Y = ReLU(A @ B)
# ==========================================

def matmul_relu_python(A, B, M, K, N):
    """
    纯 Python 实现的矩阵乘法与 ReLU 激活。
    这里故意使用原生 list 和三重 for 循环，
    完美暴露 Python 解释执行和动态类型检查的性能瓶颈。
    """
    # 初始化输出矩阵 C (M x N)
    C = [[0.0] * N for _ in range(M)]
    
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                # 乘加运算
                s += A[i][k] * B[k][j]
            # ReLU 激活：小于0截断为0
            C[i][j] = s if s > 0 else 0.0
            
    return C

if __name__ == "__main__":
    # 1. 设定矩阵维度 (这个规模下纯 Python 大概需要跑几秒钟)
    M, K, N = 256, 256, 256
    
    print(f"[*] 正在准备数据：生成维度为 {M}x{K} 和 {K}x{N} 的随机矩阵...")
    
    # 2. 生成纯 Python 二维列表
    A_py = [[random.random() for _ in range(K)] for _ in range(M)]
    B_py = [[random.random() for _ in range(N)] for _ in range(K)]
    
    print("[*] 开始执行纯 Python 版本的算子 (请耐心等待)...")
    
    # 3. 记录时间并执行
    start_time = time.perf_counter()
    C_py = matmul_relu_python(A_py, B_py, M, K, N)
    end_time = time.perf_counter()
    
    # 4. 打印基准测试结果
    print("-" * 40)
    print(f"🚀 纯 Python 耗时: {end_time - start_time:.4f} 秒")
    print(f"💡 计算结果示例 C[0][0]: {C_py[0][0]:.4f}")
    print("-" * 40)

import ctypes
import os

# ==========================================
# 模块二：Rust 底层 C 接口库 (ctypes 绑定)
# 对应课件 L3: 静态绑定与 Native Code 交互
# ==========================================

# 1. 找到编译好的动态库路径 (Windows 下是 .dll)
lib_path = os.path.join(os.path.dirname(__file__), "rs_kernel", "target", "release", "rs_kernel.dll")

# 2. 加载动态库 [cite: 1174, 1175]
rs_lib = ctypes.cdll.LoadLibrary(lib_path)

# 3. 显式指定函数的参数类型 (argtypes) [cite: 1168, 1178-1181]
# 依次对应 Rust 里的: a_ptr, b_ptr, c_ptr, m, k, n
rs_lib.matmul_relu_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t
]
# 指定返回类型为空
rs_lib.matmul_relu_c.restype = None

print("\n[*] 正在转换数据格式 (Python List -> 连续内存 C Array)...")
# 本地代码直接在原生机器指令上执行，需要连续内存区域 [cite: 1169, 1170]
# 将纯 Python 的二维列表展平，并转换为 ctypes 的连续内存数组
FlatArrayA = ctypes.c_double * (M * K)
FlatArrayB = ctypes.c_double * (K * N)
FlatArrayC = ctypes.c_double * (M * N)

a_flat = FlatArrayA(*[val for row in A_py for val in row])
b_flat = FlatArrayB(*[val for row in B_py for val in row])
c_flat = FlatArrayC(*[0.0] * (M * N))

print("[*] 开始执行 Rust (ctypes) 版本的算子...")

# 4. 记录时间并调用底层函数
start_time_rs = time.perf_counter()
rs_lib.matmul_relu_c(a_flat, b_flat, c_flat, M, K, N)
end_time_rs = time.perf_counter()

# 5. 打印对比结果
print("-" * 40)
print(f"🚀 Rust (ctypes) 耗时: {end_time_rs - start_time_rs:.4f} 秒")
print(f"💡 计算结果示例 C[0][0]: {c_flat[0]:.4f}")
print(f"🔥 加速比: {(end_time - start_time) / (end_time_rs - start_time_rs):.2f} 倍")
print("-" * 40)
