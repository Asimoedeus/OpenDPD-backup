import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import resample
import matplotlib
matplotlib.use('TkAgg')

# 加载数据
def load_iq_data(file_path):
    data = pd.read_csv(file_path)
    return data['I'].values + 1j * data['Q'].values


# OFDM解调函数
def demodulate_ofdm(iq_samples, fft_size=256, cp_length=64, sample_rate=800e6, bandwidth=200e6):
    """
    解调OFDM信号获取QAM符号

    参数:
    iq_samples: 输入I/Q采样点
    fft_size: FFT大小 (根据带宽和采样率计算)
    cp_length: 循环前缀长度
    sample_rate: 采样率 (800 MHz)
    bandwidth: 信号带宽 (200 MHz)

    返回:
    qam_symbols: 解调后的QAM符号
    """
    # 1. 计算每符号长度
    symbol_length = fft_size + cp_length
    num_symbols = len(iq_samples) // symbol_length

    if num_symbols == 0:
        raise ValueError(f"样本不足: 需要至少 {symbol_length} 个样本, 但只有 {len(iq_samples)} 个")

    # 2. 截取完整OFDM符号
    trimmed = iq_samples[:num_symbols * symbol_length]

    # 3. 重塑为符号矩阵
    symbols_matrix = trimmed.reshape(num_symbols, symbol_length)

    # 4. 移除循环前缀
    symbols_no_cp = symbols_matrix[:, cp_length:]

    # 5. 执行FFT
    qam_symbols = np.zeros((num_symbols, fft_size), dtype=complex)
    for i in range(num_symbols):
        qam_symbols[i] = fft(symbols_no_cp[i])

    # 6. 提取有效子载波 (去除保护带)
    active_subcarriers = int(fft_size * bandwidth / sample_rate)
    start_idx = (fft_size - active_subcarriers) // 2
    end_idx = start_idx + active_subcarriers

    return qam_symbols[:, start_idx:end_idx].flatten()


# 主程序
if __name__ == "__main__":
    # 加载数据
    input_iq = load_iq_data('datasets/DPA_200MHz/train_input.csv')
    output_iq = load_iq_data('datasets/DPA_200MHz/train_output.csv')

    # 设置OFDM参数 (根据论文: 200MHz OFDM, 800MHz采样率)
    fft_size = 512  # FFT大小
    cp_length = 128  # 循环前缀长度 (FFT大小的25%)
    sample_rate = 800e6  # 800 MHz采样率
    bandwidth = 200e6  # 200 MHz带宽

    print(f"input signal length: {len(input_iq)} samples")
    print(f"output signal length: {len(output_iq)} samples")

    # 解调输入信号
    input_qam = demodulate_ofdm(
        input_iq,
        fft_size=fft_size,
        cp_length=cp_length,
        sample_rate=sample_rate,
        bandwidth=bandwidth
    )

    # 解调输出信号
    output_qam = demodulate_ofdm(
        output_iq,
        fft_size=fft_size,
        cp_length=cp_length,
        sample_rate=sample_rate,
        bandwidth=bandwidth
    )

    print(f"got {len(input_qam)} input QAM symbols from demodulation")
    print(f"got {len(output_qam)} output QAM symbols from demodulation")

    # 绘制对比图
    plt.figure(figsize=(15, 8))

    # 输入信号星座图 (预失真)
    plt.subplot(121)
    plt.scatter(np.real(input_qam), np.imag(input_qam), s=1, alpha=0.5)
    plt.title("Try demodulation")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.axis('equal')

    # 输出信号星座图 (功放失真)
    plt.subplot(122)
    plt.scatter(np.real(output_qam), np.imag(output_qam), s=1, alpha=0.5, c='red')
    plt.title("Constellation")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig("qam_constellation_comparison.png", dpi=300)
    plt.show()


    # 计算并显示EVM
    def calculate_evm(received, ideal):
        """计算误差向量幅度 (EVM)"""
        evm = np.sqrt(np.mean(np.abs(received - ideal) ** 2) / np.sqrt(np.mean(np.abs(ideal) ** 2)))
        return 20 * np.log10(evm)


    evm_input = calculate_evm(input_qam, input_qam)  # 应为0 (理想参考)
    evm_output = calculate_evm(output_qam, input_qam)

    print("\n性能指标:")
    print(f"输入信号EVM: {evm_input:.2f} dB")
    print(f"输出信号EVM: {evm_output:.2f} dB")
    print(f"EVM恶化: {evm_output - evm_input:.2f} dB")