// 告诉 Rust 编译器不要改变函数名（no_mangle），并且按照 C 语言的标准（extern "C"）导出接口
#[unsafe(no_mangle)]
pub extern "C" fn matmul_relu_c(
    a_ptr: *const f64, 
    b_ptr: *const f64, 
    c_ptr: *mut f64, 
    m: usize, 
    k: usize, 
    n: usize
) {
    // 因为涉及裸指针（Raw Pointer）的直接内存操作，必须放在 unsafe 块中
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    // 把一维数组当成二维矩阵来寻址
                    let a_val = *a_ptr.add(i * k + p);
                    let b_val = *b_ptr.add(p * n + j);
                    s += a_val * b_val;
                }
                // ReLU 激活
                let out_val = if s > 0.0 { s } else { 0.0 };
                // 将结果写回输出矩阵的内存位置
                *c_ptr.add(i * n + j) = out_val;
            }
        }
    }
}
