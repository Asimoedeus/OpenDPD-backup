# Memory‑Polynomial Power Amplifier (PA_MP)
import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    def __init__(self, coeff_mat=None):
        """
        coeff_mat : 2‑D numpy array, shape (P, M)
            Row p, col m holds a_{p+1,m}
            If None → 用内置示例系数
        """
        gr.sync_block.__init__(self,
            name = "PA_model_Memory_Polynomial",
            in_sig  = [np.complex64],
            out_sig = [np.complex64])

        # 默认 5 阶、记忆 3
        if coeff_mat is None:
            coeff_mat = np.array([
                [ 1.00+0j , -0.02+0j , 0.00+0j ],   # p=1
                [ 0.00+0j ,  0.00+0j , 0.00+0j ],   # p=2 (unused)
                [-0.10+0j , -0.05+0j , 0.00+0j ],   # p=3
                [ 0.00+0j ,  0.00+0j , 0.00+0j ],   # p=4
                [ 0.02+0j ,  0.00+0j , 0.00+0j ],   # p=5
            ], dtype=np.complex64)
        self.coeff = coeff_mat
        self.P, self.M = self.coeff.shape
        # 开辟记忆缓冲：每列一个滞后分量
        self.mem = np.zeros((self.M,), dtype=np.complex64)

    def work(self, input_items, output_items):
        x = input_items[0]
        y = np.empty_like(x)

        for n, xn in enumerate(x):
            # 滚动缓冲区：mem[0] 是当前样本
            self.mem[1:] = self.mem[:-1]
            self.mem[0]  = xn
            acc = 0+0j
            for p in range(self.P):
                # |x|^{p}（p 从 0 开始对应一阶项）
                abs_pow = np.abs(self.mem) ** p
                acc += np.sum(self.coeff[p, :] * self.mem * abs_pow)
            y[n] = acc
        output_items[0][:] = y
        return len(y)
