use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use ndarray::Array2;

#[pyfunction]
fn matmul_relu_pyo3<'py>(
    py: Python<'py>,
    a_py: PyReadonlyArray2<f64>, // 接收 Python 传来的 NumPy 二维数组 (只读)
    b_py: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    // 1. 将 Python 的 NumPy 数组转换为 Rust 的 ndarray 视图
    let a = a_py.as_array();
    let b = b_py.as_array();

    // 2. 获取矩阵维度
    let m = a.shape()[0];
    let k = a.shape()[1];
    let k2 = b.shape()[0];
    let n = b.shape()[1];

    // 3. 安全性检查
    if k != k2 {
        return Err(PyValueError::new_err("矩阵维度不匹配，无法相乘！"));
    }

    // 4. 在 Rust 中分配输出矩阵的内存，初始化为 0
    let mut c = Array2::<f64>::zeros((m, n));

    // 5. 核心计算逻辑
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

    // 6. 将 Rust 的数组所有权转移给 Python，变成一个标准的 NumPy 数组返回
    Ok(c.into_pyarray(py))
}

// 注册 Python 模块
#[pymodule]
fn pyo3_lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul_relu_pyo3, m)?)?;
    Ok(())
}
