# Python 作为胶水语言调用 Rust 计算内核：基于 ctypes 与 PyO3 的实现、原理与性能比较
## 问题定义
在第三课`Python 与 Native Code 交互`中，我们学习了如何在python中用FFI调用预先编译好的C语言代码

同样作为编译型语言的rust理论上也可以用类似的方法被python程序调用

经过查询资料得知确实可以，但实践中更广泛的方法是用`pyo3`库结合`numpy`等依赖，处理成python可用的库`import`后直接使用

这两种方法有什么区别？性能如何？

## 先导结论
PyO3 实现在大多数情况下性能优于 Ctypes FFI实现

## 实验
作为本次探索计算任务的是`C = ReLU(A * B)`，带有ReLU激活函数的矩阵乘法

### 实现
对于两种方案，为了让cargo知道要导出为动态库，都需要在cargo.toml中指明：
```toml
[lib]
crate-type = ["cdylib"]
```

#### ctypes FFI 模式
```rust
#[unsafe(no_mangle)]
pub extern "C" fn matmul_relu_c(
    a_ptr: *const f64, 
    b_ptr: *const f64, 
    c_ptr: *mut f64, 
    m: usize, 
    k: usize, 
    n: usize
) {
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    let a_val = *a_ptr.add(i * k + p);
                    let b_val = *b_ptr.add(p * n + j);
                    s += a_val * b_val;
                }
                // ReLU 激活
                let out_val = if s > 0.0 { s } else { 0.0 };
                // 将结果写回输出矩阵
                *c_ptr.add(i * n + j) = out_val;
            }
        }
    }
}
```
相比于正常的rust函数，这个函数的不同之处在于：

1. `unsafe 与裸指针` 和其他FFI一样，python直接按照C语言约定将参数放入寄存器，然后rust程序直接访问内存，中间省略了rust的抽象，所以显然是unsafe的
2. `extern C` 用于规定函数接口，按C 语言调用约定
3. #[no_mangle] 保留符号信息，为了便于调试

在python一侧
```python
# 首先加载编译后的动态链接库
lib_path = 
rs_lib = ctypes.cdll.LoadLibrary(lib_path)
# 明确函数签名，参数与返回值
rs_lib.matmul_relu_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t
]
rs_lib.matmul_relu_c.restype = None

# 准备参数，处理成连续的，提取指针
A = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
B = np.array([[2.0, 1.0], [-1.0, 3.0]], dtype=np.float64)

A_c = np.ascontiguousarray(A, dtype=np.float64)
B_c = np.ascontiguousarray(B, dtype=np.float64)
C_ctypes = np.zeros((M, N), dtype=np.float64)

a_ptr = A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
b_ptr = B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
c_ptr = C_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# 完成调用
rs_lib.matmul_relu_c(a_ptr, b_ptr, c_ptr, M, K, N)
```

加载编译后的动态链接库，明确参数类型，处理参数，最后完成调用

####  PyO3 + numpy + ndarray模式
这三个常用依赖中，`PyO3`将rust代码编译到python可import的依赖，`ndarray`是rust生态下的numpy（N维数组计算库），本身与python无关，`numpy`则用于将python的numpy数组“零拷贝”映射到`ndarray`的类型

```toml
[dependencies]
pyo3 = {version = "0.21", feature = ["extension-moduel"]}
numpy = "0.21"
ndarray = "0.15.6"

[lib]
name = "pyo3_lib"
crate-type = ["cdylib"]
```

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use ndarray::Array2;

#[pyfunction]
fn matmul_relu_pyo3<'py>(
    py: Python<'py>,
    a_py: PyReadonlyArray2<f64>, 
    b_py: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let a = a_py.as_array();
    let b = b_py.as_array();

    let m = a.shape()[0];
    let k = a.shape()[1];
    let k2 = b.shape()[0];
    let n = b.shape()[1];

    if k != k2 {
        return Err(PyValueError::new_err("矩阵维度不匹配，无法相乘！"));
    }

    // 在 Rust 中分配输出矩阵的内存，初始化为 0
    let mut c = Array2::<f64>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                // 安全的索引访问，如果越界 Rust 会 panic，而 PyO3 会将 panic 拦截为 Python 异常
                s += a[[i, p]] * b[[p, j]];
            }
            // ReLU 激活
            c[[i, j]] = if s > 0.0 { s } else { 0.0 };
        }
    }

    //将 Rust 的数组所有权转移给 Python，变成一个标准的 NumPy 数组返回
    Ok(c.into_pyarray(py))
}

// 注册 Python 模块
#[pymodule]
fn pyo3_lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul_relu_pyo3, m)?)?;
    Ok(())
}
```
对比ctypes实现，最重要的区别是没有了显式`unsafe`的声明，同时结果矩阵不需要python程序提前分配，改由rust程序内部获取

`#[pyfunction]`宏负责将自定义的函数转成了python可用的函数

`pyo3`库用一个Python token提供全局的Python解释器API、证明持有GIL、作为生命周期标识.同时用`PyResult`优雅地处理错误

`numpy`依赖提供了PyReadonlyArray2类型

`PyReadonlyArray2`是一个二维数组，`as_array()`方法将其转化为numpy库易处理的n维数组,所以对矩阵内容的访问可以用`a[[i,j]]`的形式，不用自己计算索引

`#[pymodule]`宏声明，添加刚才定义的函数，在python中`import`时触发

```python
import pyo3_lib
A = np.random.rand(size, size).astype(np.float64)
B = np.random.rand(size, size).astype(np.float64)
pyo3_lib.matmul_relu_pyo3(A, B)
```

由于类型是PyO3主动适应，python端的使用就变得异常简单

### 性能测试
分别运行 `cargo build --release` ，
```python
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

    # 使用timeit库来计时
    # --- 测试 PyO3 ---
    pyo3_time = timeit.timeit(lambda: pyo3_lib.matmul_relu_pyo3(A, B), number=iterations)
    print(f"  -> PyO3 耗时:   {pyo3_time:.5f} 秒")

    # --- 测试 ctypes ---
    ctypes_time = timeit.timeit(lambda: rs_lib.matmul_relu_c(a_ptr, b_ptr, c_ptr, size, size, size), number=iterations)
    print(f"  -> ctypes 耗时: {ctypes_time:.5f} 秒")
```

```bash
(pip_venv) root@d548a10c451e /workspace2# python c2.py

[*] 开始压测: 矩阵维度 2x2, 循环 1000000 次
  -> PyO3 耗时:   1.28316 秒
  -> ctypes 耗时: 3.37028 秒
  => 倍率: PyO3 是 ctypes 的 0.38 倍开销


[*] 开始压测: 矩阵维度 200x200, 循环 1000 次
  -> PyO3 耗时:   11.37054 秒
  -> ctypes 耗时: 10.92190 秒
  => 倍率: PyO3 是 ctypes 的 1.04 倍开销
```

发现在以调用开销为主的小矩阵计算情境下，PyO3版本具有明显优势，而当任务以计算为主，二者几乎没有差距

### 原理分析 
直观上理解，Ctypes更加底层，而PyO3做的抽象更多，前者的性能理应比后者高才对，那么这两个过程中究竟发生了什么？

首先用gdb动调观察

#### Ctypes
通过`set breakpoint pending on` `b matmul_relu_c` `c` 停在rust函数入口

`bt`查看栈：
```bash
pwndbg> bt
#0 rs_kernel::matmul_relu_c (a_ptr=0x2c34f460, b_ptr=0x2c3a61d0,
    c_ptr=0x2c3a6290, m=2, k=2, n=2) at src/lib.rs:13
#1  0x00007533ba546b16 in ?? () from /lib/x86_64-linux-gnu/libffi.so.8
#2  0x00007533ba5433ef in ?? () from /lib/x86_64-linux-gnu/libffi.so.8
#3  0x00007533ba5460be in ffi_call ()
   from /lib/x86_64-linux-gnu/libffi.so.8
#4  0x00007533ba56411c in ?? ()
   from /usr/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so
#5  0x00007533ba55f2af in ?? ()
   from /usr/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so
#6  0x00000000005492f5 in _PyObject_MakeTpCall ()
#7  0x00000000005d68bf in _PyEval_EvalFrameDefault ()
#8  0x00000000005d4dab in PyEval_EvalCode ()
#9  0x0000000000607fc2 in ?? ()
#10 0x00000000006b4393 in ?? ()
#11 0x00000000006b40fa in _PyRun_SimpleFileObject ()
#12 0x00000000006b3f2f in _PyRun_AnyFileObject ()
#13 0x00000000006bbf45 in Py_RunMain ()
#14 0x00000000006bba2d in Py_BytesMain ()
#15 0x00007533bcc511ca in __libc_start_call_main (
    main=main@entry=0x518bd0, argc=argc@entry=2,
    argv=argv@entry=0x7ffc4e144248)
    at ../sysdeps/nptl/libc_start_call_main.h:58
#16 0x00007533bcc5128b in __libc_start_main_impl (main=0x518bd0,
    argc=2, argv=0x7ffc4e144248, init=<optimized out>,
    fini=<optimized out>, rtld_fini=<optimized out>,
    stack_end=0x7ffc4e144238) at ../csu/libc-start.c:360
#17 0x0000000000656a35 in _start ()
```
可以看到从python到rust中间有一层`ffi_call`，最后给matmul_relu_c函数提供的是三个指针和矩阵长宽信息

```bash
pwndbg> x/4fg  0x2c34f460
0x2c34f460:     1       -2
0x2c34f470:     3       4
pwndbg> x/4fg  0x2c3a61d0
0x2c3a61d0:     2       1
0x2c3a61e0:     -1      3
```

#### pyo3
通过在加载后运行`info functions matmul_relu_pyo3`获取准确符号信息

通过`set breakpoint pending on` `b 'pyo3_lib::matmul_relu_pyo3'` `c` 停在rust函数入口

`bt`后得到：
```bash
pwndbg> bt
#0  pyo3_lib::matmul_relu_pyo3 (a_py=..., b_py=...) at src/lib.rs:13
#1  0x0000789938d3cac0 in pyo3_lib::__pyfunction_matmul_relu_pyo3 (
    _slf=0x789972697880, _args=0x789972c85078, _nargs=2, _kwnames=0x0)
    at src/lib.rs:6
#2  0x0000789938d3eaa3 in pyo3::impl_::trampoline::fastcall_with_keywords::{closure#0} ()
#3  0x0000789938d3e9fc in pyo3::impl_::trampoline::trampoline::{closure#0}<pyo3::impl_::trampoline::fastcall_with_keywords::{closure_env#0}, *mut pyo3_ffi::object::PyObject> ()
#4  0x0000789938d3d84a in std::panicking::catch_unwind::do_call<pyo3::impl_::trampoline::trampoline::{closure_env#0}<pyo3::impl_::trampoline::fastcall_with_keywords::{closure_env#0}, *mut pyo3_ffi::object::PyObject>, core::result::Result<*mut pyo3_ffi::object::PyObject, pyo3::err::PyErr>> (data=0x7ffc85ce2f08)
#5  0x0000789938d3dc3b in __rust_try () from /workspace3/pyo3_lib.so
#6  0x0000789938d3daa9 in std::panicking::catch_unwind<core::result::Result<*mut pyo3_ffi::object::PyObject, pyo3::err::PyErr>, pyo3::impl_::trampoline::trampoline::{closure_env#0}<pyo3::impl_::trampoline::fastcall_with_keywords::{closure_env#0}, *mut pyo3_ffi::object::PyObject>> (
    f=...)
#7  std::panic::catch_unwind<pyo3::impl_::trampoline::trampoline::{closure_env#0}<pyo3::impl_::trampoline::fastcall_with_keywords::{closure_env#0}, *mut pyo3_ffi::object::PyObject>, core::result::Result<*mut pyo3_ffi::object::PyObject, pyo3::err::PyErr>> (f=...)
#8  0x0000789938d3e91f in pyo3::impl_::trampoline::trampoline<pyo3::impl_::trampoline::fastcall_with_keywords::{closure_env#0}, *mut pyo3_ffi::object::PyObject> (body=...)
#9  0x0000789938d472f9 in pyo3::impl_::trampoline::fastcall_with_keywords (slf=0x789972697880, args=0x789972c85078, nargs=2, kwnames=0x0,
    f=0x789938d3c730 <pyo3_lib::__pyfunction_matmul_relu_pyo3>)
#10 0x0000789938d3ccc4 in pyo3_lib::{impl#0}::_PYO3_DEF::trampoline (
    _slf=0x789972697880, _args=0x789972c85078, _nargs=2, _kwnames=0x0)
    at src/lib.rs:6
#11 0x00000000005818ed in ?? ()
#12 0x0000000000549cf5 in PyObject_Vectorcall ()
#13 0x00000000005d68bf in _PyEyval_EvalFrameDefault ()
#14 0x00000000005d4dab in PyEval_EvalCode ()
#15 0x0000000000607fc2 in ?? ()
#16 0x00000000006b4393 in ?? ()
#17 0x00000000006b40fa in _PyRun_SimpleFileObject ()
#18 0x00000000006b3f2f in _PyRun_AnyFileObject ()
#19 0x00000000006bbf45 in Py_RunMain ()
#20 0x00000000006bba2d in Py_BytesMain ()
#21 0x000078997296c1ca in __libc_start_call_main (
    main=main@entry=0x518bd0, argc=argc@entry=2,
    argv=argv@entry=0x7ffc85ce36f8)
    at ../sysdeps/nptl/libc_start_call_main.h:58
#22 0x000078997296c28b in __libc_start_main_impl (main=0x518bd0,
    argc=2, argv=0x7ffc85ce36f8, init=<optimized out>,
    fini=<optimized out>, rtld_fini=<optimized out>,
    stack_end=0x7ffc85ce36e8) at ../csu/libc-start.c:360
#23 0x0000000000656a35 in _start ()
```

观察参数：
```bash
pwndbg> info args
a_py = numpy::borrow::PyReadonlyArray<f64, ndarray::dimension::dim::Dim<[usize; 2]>> {
  array: pyo3::instance::Bound<numpy::array::PyArray<f64, ndarray::dimension::dim::Dim<[usize; 2]>>> (
    pyo3::marker::Python (
      core::marker::PhantomData<(&pyo3::gil::GILGuard, pyo3::impl_::not_send::NotSend)>
    ),
    core::mem::manually_drop::ManuallyDrop<pyo3::instance::Py<numpy::array::PyArray<f64, ndarray::dimension::dim::Dim<[usize; 2]>>>> {
      value: pyo3::instance::Py<numpy::array::PyArray<f64, ndarray::dimension::dim::Dim<[usize; 2]>>> (
        core::ptr::non_null::NonNull<pyo3_ffi::object::PyObject> {
          pointer: 0x7e857a0f1e90
        },
        core::marker::PhantomData<numpy::array::PyArray<f64, ndarray::dimension::dim::Dim<[usize; 2]>>>
      )
    }
  )
}
b_py = numpy::borrow::PyReadonlyArray<f64, ndarray::dimension::dim::Dim<[usize; 2]>> {
  array: pyo3::instance::Bound<numpy::array::PyArray<f64, ndarray::dimension::dim::Dim<[usize; 2]>>> (
    pyo3::marker::Python (
      core::marker::PhantomData<(&pyo3::gil::GILGuard, pyo3::impl_::not_send::NotSend)>
    ),
    core::mem::manually_drop::ManuallyDrop<pyo3::instance::Py<numpy::array::PyArray<f64, ndarray::dimension::dim::Dim<[usize; 2]>>>> {
      value: pyo3::instance::Py<numpy::array::PyArray<f64, ndarray::dimension::dim::Dim<[usize; 2]>>> (
        core::ptr::non_null::NonNull<pyo3_ffi::object::PyObject> {
          pointer: 0x7e857a0f1f50
        },
        core::marker::PhantomData<numpy::array::PyArray<f64, ndarray::dimension::dim::Dim<[usize; 2]>>>
      )
    }
  )
}
```

查看指针：
```bash
pwndbg> x/16gx 0x7e857a0f1e90
0x7e857a0f1e90: 0x0000000000000003      0x00007e85b3951a00
0x7e857a0f1ea0: 0x000000003b1bc460      0x0000000000000002
0x7e857a0f1eb0: 0x000000003b335220      0x000000003b335230
0x7e857a0f1ec0: 0x0000000000000000      0x00007e85b3947180
0x7e857a0f1ed0: 0x0000000000000505      0x0000000000000000
```
按照ndarray的标准，
```c
typedef struct PyArrayObject {
    PyObject_HEAD           /* 宏展开后就是：引用计数(ob_refcnt) 和 类型指针(ob_type) */
    char *data;             /* 指向真实物理矩阵数据的指针  */
    int nd;                 /* 维度的数量  e(2) */
    npy_intp *dimensions;   /* 形状数组指针 (Shape) */
    npy_intp *strides;      /* 步长数组指针 (Strides) */
    PyObject *base;         /* 基对象指针 */
    PyArray_Descr *descr;   /* 数据类型描述符 (float64) */
    int flags;              /* 内存连续性等标志位 */
    PyObject *weakreflist;
} PyArrayObject;
```

提取数据：
```bash
pwndbg> x/4fg 0x000000003b1bc460
0x3b1bc460:     1       -2
0x3b1bc470:     3       4
```
#### 总结
可以发现PyO3版本的调用栈确实比Ctypes的更深，包括PyO3的类型处理层、rust的错误处理层等等

两版从`_PyEval_EvalFrameDefault`开始不同，这是第二课里python解释器执行python字节码的行为

Pyo3最终提供给矩阵乘法的数据类型是经过复杂包装的，而ctypes的是直接提供指针

但是依据这里的信息无法解释性能问题

### Flame Graph分析

运行`PYTHONPERFSUPPORT=1 perf record -e cpu-clock -F 999 --call-graph dwarf -- python3 v2.py`再`perf script | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > benchmark.svg`得到若干次小矩阵计算的火焰图

pyo3版调用栈更深，所以高度更高，ctypes版用时更长所以更深，符合预期，二者从 _PyEyval_EvalFrameDefault开始分开，也和刚才的调用栈一致

ctypes 主要有五个耗时操作，PyObject_CallOneArg 38%，ffi_call 7%，gc相关 6.5%，lib_ffi.so 3.6%，线程相关2.5%（占总时间）

PyObject_CallOneArg 每次调用 rs_lib.matmul_relu_c(a_ptr, b_ptr, c_ptr, M, K, N) 时，ctypes 并不能直接把 Python 的整数或指针扔给底层。它必须根据之前定义的 argtypes，在运行时将这些 Python 对象转换成对应的 C 类型封装对象。
所以这 38% 的时间，绝大部分花在了高频地实例化 ctypes 包装对象上

pyo3 (18.65%) 的第一部分开销是`pyo3::impl_::extract_argument::extract_argument` 提取参数，5.55%，rust-numpy 库在每次创建 PyReadonlyArray 时，都会动态地把这个数组的地址注册到一个全局/线程局部的哈希表里加锁，所以可以看见大量的hashbrown占比

于是对应的`<numpy::borrow::PyReadonlyArray<T,D> as core::ops::drop::Drop>::drop`销毁也有很大开销 3.56%

结果矩阵是在rust部分里分配的，所以有分配内存的开销 0.88%，将结果用`numpy::convert::IntoPyArray::into_pyarray`转为python可处理的结构 4.49%

额外还有 GIL(global interpreter lock) 2%，全局的Python解释器，在多线程中解决并发错误，测试代码中并没有做多线程支持

## 进一步探索
通过对比可知，ctypes版本的主要开销在于用python处理c代码可用的类型，pyo3则是可以复用python类型，只做提取，同时一大部分开销在于为实现rust需要的安全性而做的包装

那么如果在pyo3的基础上放弃安全检查使用裸指针规避类型开销，理论上就可以获得更高的性能：

```rust
#[pyfunction]
#[pyo3(signature = (a_ptr, b_ptr, c_ptr, m, k, n))]
fn matmul_relu_frankenstein(
    a_ptr: usize, 
    b_ptr: usize,
    c_ptr: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let a = a_ptr as *const f64;
    let b = b_ptr as *const f64;
    let c = c_ptr as *mut f64;

    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    let a_val = *a.add(i * k + p);
                    let b_val = *b.add(p * n + j);
                    s += a_val * b_val;
                }
                let out_val = if s > 0.0 { s } else { 0.0 };
                *c.add(i * n + j) = out_val;
            }
        }
    }
}
```

实验也证明了这一点：
```bash
[*] 开始压测: 矩阵维度 2x2, 循环 1000000 次
  -> PyO3  耗时:   0.97994 秒
  -> ctypes  耗时: 2.31750 秒
  -> PyO3 (改) 耗时:   0.28359 秒
```
观察火焰图可以发现主要开销只剩下`GIL`和extract_argument`

## 结论总结
通过对“单层带有 ReLU 激活的矩阵乘法”算子在纯 Python、Ctypes FFI 以及 PyO3 原生扩展三种模式下的实现与深度性能剖析，本次探究得出以下核心结论与工程启示：

1. FFI 边界的“过路费”是小任务性能的核心瓶颈
在极小矩阵的高频调用场景下，跨语言调用的固定开销（FFI Overhead）被无限放大。

火焰图清晰表明，Ctypes 的主要性能损耗（占比逾 38%）发生在 PyObject_CallOneArg 与 libffi 的动态类型转换上。Python 必须在运行时将高级对象高频实例化并降级为 C 指针，这条漫长的调用链路严重拖慢了执行效率。

PyO3 的“编译期红利”与“安全税”：PyO3 通过宏在编译期直接生成了符合 Python C API 的原生扩展，省去了运行时的动态翻译，因此在基础调用上显著快于 Ctypes。然而，为了保证 Rust 严苛的内存安全，rust-numpy 在提取 PyReadonlyArray 时引入了全局/线程局部的内存借用校验（Hash 表操作）以及 GIL 管理，这构成了 PyO3 安全模式下的主要开销。

2. 极致性能与内存安全的工程权衡
“Frankenstein（缝合怪）”变种实验揭示了跨语言调用的性能天花板：当我们在 PyO3 中完全剥离类型安全检查，退化为直接传递 usize 物理内存地址并使用 unsafe 裸指针计算时，耗时被极致压缩（达到 Ctypes 的 1/8）。
但这在实际 AI 工业落地中并不值得提倡。现代 AI 基础设施（如 Hugging Face 的底层组件）之所以全面转向 Rust (PyO3) 而非停留在 C 语言，正是因为愿意支付极小的“安全税”（约 5% 的校验开销），来换取运行时免于 Segmentation Fault 的绝对内存安全与极佳的面向对象开发体验。

3. 对 AI 软件底层工程架构的启示
避免高频跨界，传递“整块数据”：无论是 Ctypes 还是 PyO3，都不应在 Python 的 for 循环中去高频调用原生算子。正确的架构范式是“Python 负责计算图的顶层编排，Rust/C++ 负责底层重负载运算”，一次性跨界传递大维度的张量（Tensor）或矩阵，用计算的绝对耗时来摊薄 FFI 边界的固定成本。

原生扩展模块是未来的最优解：相比于手工处理指针、极易出错且性能并不占优的 Ctypes 方案，采用 PyO3 等原生扩展绑定框架，在开发成本、类型安全、接口复用度以及运行性能上均实现了降维打击，是构建下一代 AI 算子库的最佳实践路线。


