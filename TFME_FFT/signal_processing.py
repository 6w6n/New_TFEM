import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hann, flattop

# =====================================================================
# 1. 不叠加傅里叶变换 (长序列直接 FFT)
# 对应功能菜单：2
# =====================================================================
def fft_no_stack(timeseries, sample_rate):
    """
    对整段输入的时间序列直接进行 FFT 变换。

    参数:
        timeseries: 一维或二维数组 (n_chan, n_points)
        sample_rate: 采样率 (Hz)
    返回:
        xf: 频率轴数组
        yf: 对应的复数频谱矩阵
    """
    # 获取序列长度 (取最后一个维度的长度)
    n_sam = timeseries.shape[-1]

    # 执行 FFT
    yf_raw = fft(timeseries, axis=-1)

    # 计算对应的频率轴
    xf = fftfreq(n_sam, 1.0 / sample_rate)

    # FFT 幅值缩放 (严格对齐原版 Fortran 的 / N * 2 逻辑)
    yf = yf_raw * 2.0 / n_sam

    return xf, yf


# =====================================================================
# 2. 叠加傅里叶变换 - 时域 (Time-domain Stacking)
# 对应功能菜单：3
# =====================================================================
def time_domain_stacking(time_series, cyc_len, cyc_num):
    """
    时域叠加算法 (Time-Domain Stacking / Synchronous Averaging)

    参数:
        time_series: 1D NumPy 数组，原始长序列波形数据
        cyc_len: 整数，单个完整循环的采样点数 (Cycle Length)
        cyc_num: 整数，循环的总个数 (Cycle Number)

    返回:
        stacked_wave: 1D NumPy 数组，长度为 cyc_len 的叠加后单周期波形
    """
    expected_len = cyc_len * cyc_num

    # 截取有效长度（丢弃末尾可能多余的不完整碎片点）
    valid_data = time_series[:expected_len]

    # 【核心操作】将 1D 数组重塑为 2D 矩阵：形状为 (cyc_num, cyc_len)
    # 这相当于把波形按照周期长度一段段切开并对齐
    matrix = valid_data.reshape((cyc_num, cyc_len))

    # 沿列方向 (axis=0) 求算术平均，得到叠加后的波形
    stacked_wave = np.mean(matrix, axis=0)

    return stacked_wave


def fft_short(stacked_wave, sample_rate):
    """
    对叠加后的单周期波形进行快速傅里叶变换 (FFT)
    """
    n = len(stacked_wave)
    # 计算频率轴
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    # 计算复数频谱
    yf = np.fft.fft(stacked_wave)

    return freqs, yf


# =====================================================================
# 3. 叠加傅里叶变换 - 频域 (Frequency-domain Stacking)
# 对应功能菜单：4
# =====================================================================
def fft_freq_stacking(timeseries, sample_rate, cyc_len, cyc_num):
    """
    先对每一个单独的循环周期分别做 FFT，得到 N 个频谱图。
    然后在频域上对这 N 个复数频谱图进行叠加求平均。

    参数:
        timeseries: 二维数组 (n_chan, n_total_points)
        sample_rate: 采样率 (Hz)
        cyc_len: 单个周期的采样点数
        cyc_num: 循环的次数
    返回:
        xf: 频率轴数组
        yf_stacked: 频域平均后的复数频谱矩阵
    """
    n_chan = timeseries.shape[0]
    expected_len = cyc_len * cyc_num

    # 截取
    valid_ts = timeseries[:, :expected_len]

    # 同样切分成: (通道数, 循环数, 单周期点数)
    reshaped_ts = valid_ts.reshape((n_chan, cyc_num, cyc_len))

    # [核心1] 针对每一个单独的循环独立做 FFT (沿着最后的点数维度 axis=-1)
    yf_all_cycles = fft(reshaped_ts, axis=-1)

    # [核心2] 在频率域上沿着循环维度 (axis=1) 把所有复数相加求平均
    yf_stacked_raw = np.mean(yf_all_cycles, axis=1)

    # 计算频率轴
    xf = fftfreq(cyc_len, 1.0 / sample_rate)

    # 缩放
    yf_stacked = yf_stacked_raw * 2.0 / cyc_len

    return xf, yf_stacked


# =====================================================================
# 4. 辅助工具：提取目标频点的复数 (用于最后算视电阻率)
# =====================================================================
def extract_target_frequency(xf, yf_matrix, target_freq):
    """
    从 FFT 计算出的全频谱中，精准提取主频(目标频率)的那一个复数结果。

    参数:
        xf: 频率轴数组
        yf_matrix: 复数频谱矩阵 (n_chan, freq_points)
        target_freq: 目标频率 (Hz)
    返回:
        target_complex: 各通道在目标频率下的复数数组 (含有实部和虚部)
    """
    # 寻找距离目标频率最近的索引点
    idx = np.argmin(np.abs(xf - target_freq))

    # 返回该索引对应的所有通道的复数
    target_complex = yf_matrix[:, idx]
    return target_complex


def long_fft_with_window(time_series, sample_rate, window_type='hann'):
    """
    对长序列原始波形直接加窗并做 FFT
    """
    N = len(time_series)

    # 1. 生成窗函数 (默认使用汉宁窗 Hann，抗泄露能力强)
    if window_type == 'hann':
        window = hann(N)
    elif window_type == 'flattop':
        window = flattop(N)  # 平顶窗：振幅最准，但峰会变宽
    else:
        window = np.ones(N)  # 矩形窗 (相当于不加窗)

    # 2. 将原始信号与窗函数相乘 (压平信号首尾)
    windowed_signal = time_series * window

    # 3. 窗函数能量补偿
    # 因为加窗把首尾的信号压到 0 了，整体能量会变小。
    # 比如 Hann 窗的平均值是 0.5，所以要把算出来的振幅乘以 2 补回来。
    amplitude_correction = 1.0 / np.mean(window)

    # 4. 执行快速傅里叶变换
    freqs = np.fft.fftfreq(N, d=1.0 / sample_rate)
    yf = np.fft.fft(windowed_signal)

    # 5. 直接换算成单边物理振幅
    # 标准公式: (2.0 / N) * abs(FFT) * 补偿系数
    real_amplitude = (2.0 / N) * np.abs(yf) * amplitude_correction

    return freqs, real_amplitude