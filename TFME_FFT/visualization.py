import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # 强制指定后端为 TkAgg，确保交互窗口正常弹出
from matplotlib.widgets import Slider, RadioButtons
from pathlib import Path
matplotlib.use('TkAgg')
from matplotlib.ticker import ScalarFormatter, LogLocator
# =====================================================================
# 全局绘图参数设置 (让图表看起来更专业)
# =====================================================================
import matplotlib.pyplot as plt

# 1. 设置字体为黑体，以正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 2. 【关键修复】关闭 unicode 减号，让坐标轴上的负号正常显示
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# [静态图库] 1. 时域波形可视化
# =====================================================================
def plot_waveform(data, title="Time Domain Waveform", xlabel="Sample Points", ylabel="Amplitude", num_points=None,
                  vlines=None):
    """
    画出时域的时间序列波形图。
    """
    if num_points is not None and num_points < len(data):
        plot_data = data[:num_points]
    else:
        plot_data = data

    x_axis = np.arange(len(plot_data))

    plt.figure(figsize=(12, 5))
    plt.plot(x_axis, plot_data, color='#1f77b4', linestyle='-', linewidth=1.5, marker='.', markersize=3)

    if vlines is not None:
        num_cycles = len(plot_data) // vlines
        for i in range(1, num_cycles + 1):
            plt.axvline(x=i * vlines, color='red', linestyle='--', alpha=0.6)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


# =====================================================================
# [静态图库] 2. 频域频谱可视化 (单图)
# =====================================================================
def plot_spectrum(freqs, yf_complex, title="Frequency Spectrum", max_freq=None):
    """
    画出 FFT 之后的频域振幅谱。
    """
    positive_idx = freqs >= 0
    f = freqs[positive_idx]
    amplitude = np.abs(yf_complex[positive_idx])

    if max_freq is not None:
        valid_idx = f <= max_freq
        f = f[valid_idx]
        amplitude = amplitude[valid_idx]

    plt.figure(figsize=(12, 5))
    plt.plot(f, amplitude, color='#ff7f0e', linewidth=1.5)
    plt.fill_between(f, amplitude, color='#ff7f0e', alpha=0.3)

    plt.title(title, fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# =====================================================================
# [静态图库] 3. 频域频谱对比可视化 (双图重叠)
# =====================================================================
def plot_compare_spectra(freqs, yf_complex1, yf_complex2, label1="Method 1", label2="Method 2",
                         title="Spectra Comparison", max_freq=None):
    """
    将两种不同算法（比如时域叠加 vs 频域叠加）算出的频谱画在同一张图里进行对比。
    """
    positive_idx = freqs >= 0
    f = freqs[positive_idx]

    amp1 = np.abs(yf_complex1[positive_idx])
    amp2 = np.abs(yf_complex2[positive_idx])

    if max_freq is not None:
        valid_idx = f <= max_freq
        f = f[valid_idx]
        amp1 = amp1[valid_idx]
        amp2 = amp2[valid_idx]

    plt.figure(figsize=(12, 5))
    plt.plot(f, amp1, label=label1, color='blue', linewidth=2, alpha=0.7)
    plt.plot(f, amp2, label=label2, color='red', linewidth=2, linestyle='--', alpha=0.7)

    plt.title(title, fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# =====================================================================
# [交互图库] 4. 交互式时域波形查看器
# =====================================================================
def interactive_time_viewer(ts_dir, file_stem):
    """交互式时域波形查看器 (自适应周期 + 鼠标无缝切换通道)"""
    ts_dir_path = Path(ts_dir)

    search_pattern = f"{file_stem}_#Period=*_Timeseries.txt"
    period_files = list(ts_dir_path.glob(search_pattern))
    n_periods = len(period_files)

    if n_periods == 0:
        print(f"\n❌ 严重错误: 在 {ts_dir_path} 中没有找到任何周期数据！")
        return

    print(f"\n[√] 自动检测完毕: 共找到 {n_periods} 个周期文件。")
    print(f"[>] 正在构建交互界面，请在弹出的窗口中操作...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # ⬅️ 【关键布局调整】
    # 底部留20%给滑块，左侧留20%给单选按钮
    plt.subplots_adjust(bottom=0.2, left=0.2)

    def load_period_data(p):
        path = ts_dir_path / f"{file_stem}_#Period={p:02d}_Timeseries.txt"
        if not path.exists(): return None
        return np.loadtxt(path, skiprows=2)

    # ==========================================
    # 核心升级：状态管理与自动检测通道
    # ==========================================
    initial_data = load_period_data(1)
    if initial_data is None: return

    # 自动获取数据的列数，生成通道列表 (比如 Ch 0, Ch 1...)
    num_cols = initial_data.shape[1]
    chan_labels = [f"Ch {i}" for i in range(num_cols)]

    # 维护当前状态
    state = {
        'period': 1,
        'channel': 0  # 默认显示第 0 列数据
    }

    # 绘制初始折线
    y_data_init = initial_data[:, state['channel']]
    x_data_init = np.arange(len(y_data_init))
    line, = ax.plot(x_data_init, y_data_init, lw=1, color='b')
    ax.set_title(f"Time Series: Period {state['period']} - Channel {state['channel']}")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, len(x_data_init))

    # ==========================================
    # 控件 1：底部滑动条 (控制周期)
    # ==========================================
    ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])  # [左, 下, 宽, 高]
    slider = Slider(ax_slider, 'Period', 1, n_periods, valinit=1, valstep=1)

    # ==========================================
    # 控件 2：左侧单选按钮 (控制通道)
    # ==========================================
    # 在左边画一个框用来放按钮
    ax_radio = plt.axes([0.02, 0.4, 0.12, 0.2], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(ax_radio, chan_labels, active=0)

    # ==========================================
    # 统一的重绘引擎
    # ==========================================
    def update_plot():
        data = load_period_data(state['period'])
        if data is None: return

        new_y = data[:, state['channel']]
        new_x = np.arange(len(new_y))

        line.set_xdata(new_x)
        line.set_ydata(new_y)
        ax.set_xlim(0, len(new_x))

        y_min, y_max = new_y.min(), new_y.max()
        margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        ax.set_ylim(y_min - margin, y_max + margin)

        ax.set_title(f"Time Series: Period {state['period']} - Channel {state['channel']} (Samples: {len(new_x)})")
        fig.canvas.draw_idle()

    # 滑块变动时的回调
    def on_slider_change(val):
        state['period'] = int(slider.val)
        update_plot()

    # 按钮点击时的回调
    def on_radio_change(label):
        # 从标签 "Ch 0" 中提取出数字 0
        state['channel'] = int(label.split(" ")[1])
        update_plot()

    # 绑定事件
    slider.on_changed(on_slider_change)
    radio.on_clicked(on_radio_change)

    plt.show(block=True)


# =====================================================================
# [交互图库] 5. 交互式频域频谱查看器 (显示幅度谱，已去除负频率)
# =====================================================================
def interactive_freq_viewer(fs_dir, file_stem):
    """交互式频域频谱查看器 (自适应周期数量，仅显示幅度谱，只保留正频率)"""
    fs_dir_path = Path(fs_dir)

    search_pattern = f"{file_stem}_#Period=*_Spectrum.txt"
    period_files = list(fs_dir_path.glob(search_pattern))
    n_periods = len(period_files)

    if n_periods == 0:
        print(f"\n❌ 严重错误: 在 {fs_dir_path} 中没有找到任何频谱数据！")
        return

    print(f"\n[√] 自动检测完毕: 共找到 {n_periods} 个频谱文件。")
    print(f"[>] 正在构建交互界面，请在弹出的窗口中操作...")

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)

    def load_freq_data(p):
        path = fs_dir_path / f"{file_stem}_#Period={p:02d}_Spectrum.txt"
        if not path.exists(): return None

        data = np.loadtxt(path, skiprows=1)
        raw_freq = data[:, 0]
        re = data[:, 1]
        im = data[:, 2]

        raw_amp = np.sqrt(re ** 2 + im ** 2)

        # ==========================================
        # 【修改核心】：过滤掉负频率
        # ==========================================
        pos_mask = raw_freq >= 0  # 1. 找出所有大于等于 0 的频率的布尔索引
        freq_pos = raw_freq[pos_mask]  # 2. 截取正频率
        amp_pos = raw_amp[pos_mask]  # 3. 截取对应的正频率幅度

        return freq_pos, amp_pos

    init_data = load_freq_data(1)
    if init_data is None:
        print("❌ 无法读取第 1 周期频谱数据，启动失败。")
        return

    f_init, a_init = init_data

    # 绘制幅度图 (依然使用对数Y轴)
    line, = ax.semilogy(f_init, a_init, color='r', lw=1.5)

    ax.set_title(f"Frequency Spectrum - Magnitude (Period 1)")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

    # 现在的 f_init 里最小的值就是 0 (或者接近 0 的正数)，X 轴会自动适应
    ax.set_xlim(f_init.min(), f_init.max())

    valid_a = a_init[a_init > 0]
    if len(valid_a) > 0:
        ax.set_ylim(valid_a.min() * 0.5, valid_a.max() * 2.0)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Period', 1, n_periods, valinit=1, valstep=1)

    def update(val):
        p = int(slider.val)
        data = load_freq_data(p)
        if data is None: return

        new_f, new_a = data

        line.set_xdata(new_f)
        line.set_ydata(new_a)

        ax.set_xlim(new_f.min(), new_f.max())

        valid_new_a = new_a[new_a > 0]
        if len(valid_new_a) > 0:
            ax.set_ylim(valid_new_a.min() * 0.5, valid_new_a.max() * 2.0)

        ax.set_title(f"Frequency Spectrum - Magnitude (Period {p})")

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show(block=True)


def simple_freq_channel_viewer(spectrum_filepath):
    """极简版频域查看器：专注于无缝切换不同通道的频谱"""
    path = Path(spectrum_filepath)
    if not path.exists():
        print(f"❌ 找不到文件: {path}")
        return

    # 1. 加载数据
    data = np.loadtxt(path, skiprows=1)

    # 2. 动态计算通道数
    # 第0列是频率，剩下的每2列(实部和虚部)为一个通道
    num_chans = (data.shape[1] - 1) // 2
    chan_labels = [f"Ch {i}" for i in range(num_chans)]

    # 3. 初始化绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.2, bottom=0.15)  # 左侧留出空间给按钮

    def get_amp(ch):
        re = data[:, 1 + ch * 2]
        im = data[:, 2 + ch * 2]
        return np.sqrt(re ** 2 + im ** 2)

    freqs = data[:, 0]
    current_amp = get_amp(0)  # 默认加载 Ch 0

    # 画图 (使用对数坐标)
    line, = ax.semilogy(freqs, current_amp, color='#ff7f0e', lw=1.5)

    ax.set_title(f"Frequency Spectrum - {path.name} (Ch 0)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, which="both", ls="--", alpha=0.6)

    # 动态设置坐标轴范围
    ax.set_xlim(freqs.min(), freqs.max())
    valid_amp = current_amp[current_amp > 0]
    if len(valid_amp) > 0:
        ax.set_ylim(valid_amp.min() * 0.5, valid_amp.max() * 2.0)

    # 4. 创建左侧的通道切换按钮
    ax_radio = plt.axes([0.02, 0.4, 0.12, 0.2], facecolor='lightcyan')
    radio = RadioButtons(ax_radio, chan_labels, active=0)

    # 5. 按钮点击回调函数
    def on_radio_change(label):
        # 从 "Ch 0" 中提取数字 0
        ch_idx = int(label.split(" ")[1])
        new_amp = get_amp(ch_idx)

        # 更新线条数据
        line.set_ydata(new_amp)

        # 动态更新 Y 轴范围以适应新通道的幅度
        valid_new_amp = new_amp[new_amp > 0]
        if len(valid_new_amp) > 0:
            ax.set_ylim(valid_new_amp.min() * 0.5, valid_new_amp.max() * 2.0)

        ax.set_title(f"Frequency Spectrum - {path.name} (Ch {ch_idx})")
        fig.canvas.draw_idle()

    # 绑定事件
    radio.on_clicked(on_radio_change)
    plt.show(block=True)


# =====================================================================
# 【必备辅助函数】：单点精确 DFT 算法 (专治各种频域泄露与不准)
# =====================================================================
def targeted_dft(time_series, sample_rate, target_freq):
    """直接计算指定频率下的极其精确的振幅，不受 FFT 分辨率影响"""
    N = len(time_series)
    t = np.arange(N) / sample_rate
    complex_wave = np.exp(-1j * 2 * np.pi * target_freq * t)
    X_f = np.sum(time_series * complex_wave)
    return 2.0 * np.abs(X_f) / N  # 换算为真实单峰振幅

def plot_hybrid_spectrum(time_series, sample_rate, freqs_fft, yf_fft,
                         fundamental_freq, num_harmonics=15,
                         extra_freqs=[50.0],  # ⬅️ 你可以在这里添加任何你想专门看的“其他频率”
                         title="频谱精密分析 (FFT+DFT)"):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(16, 8))

    # ==========================================
    # 步骤 1：用 FFT 画出包含“所有频率”的黑色背景线
    # ==========================================
    pos_idx = freqs_fft > 0
    f_full = freqs_fft[pos_idx]
    amp_full = np.abs(yf_fft[pos_idx])

    ax.loglog(f_full, amp_full, color='black', linewidth=0.5, alpha=0.8, label="全频段 FFT 轮廓")

    # ==========================================
    # 步骤 2：用 DFT 精准计算并标记“奇次谐波” (绿色点)
    # ==========================================
    theoretical_harmonics = [fundamental_freq * (2 * i + 1) for i in range(num_harmonics // 2 + 1)]

    for i, target_f in enumerate(theoretical_harmonics):
        if target_f > f_full.max() or target_f < f_full.min(): continue

        # 抛弃 FFT 寻峰，直接算绝对精确的 DFT 振幅
        exact_amp = targeted_dft(time_series, sample_rate, target_f)

        # 悬浮画法
        float_factor = 2.5 if i % 2 == 0 else 5.0
        dot_y = exact_amp * float_factor

        ax.plot([target_f, target_f], [exact_amp, dot_y], color='black', linewidth=0.8, zorder=4)
        ax.scatter(target_f, dot_y, color='#00FF00', s=50, edgecolors='black', zorder=5)

        harmonic_times = 2 * i + 1
        ax.text(target_f, dot_y * 1.2, f"{target_f:.1f}Hz({harmonic_times}T)\n{exact_amp:.4f}",
                rotation=30, ha='left', va='bottom', fontsize=9, color='green')

    # ==========================================
    # 步骤 3：用 DFT 精准计算并标记“其他你想看的特定频率” (比如蓝色的 50Hz)
    # ==========================================
    if extra_freqs:
        for target_f in extra_freqs:
            exact_amp = targeted_dft(time_series, sample_rate, target_f)

            # 用蓝色点区分这些“其他频率”
            dot_y = exact_amp * 4.0
            ax.plot([target_f, target_f], [exact_amp, dot_y], color='blue', linewidth=0.8, linestyle='--', zorder=4)
            ax.scatter(target_f, dot_y, color='blue', s=40, edgecolors='black', zorder=5)
            ax.text(target_f, dot_y * 1.2, f"Noise:{target_f:.1f}Hz\n{exact_amp:.4f}",
                    rotation=30, ha='left', va='bottom', fontsize=9, color='blue')

    # ==========================================
    # 工业级坐标系设置 (与之前一致)
    # ==========================================
    ax.set_xlim(1000, 0.01)

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0))
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labeltop=True)
    ax.xaxis.set_major_formatter(formatter)

    ax.grid(True, which='major', color='#a0a0a0', linestyle='-', linewidth=0.6, alpha=0.8)
    ax.grid(True, which='minor', color='#d3d3d3', linestyle=':', linewidth=0.5, alpha=0.6)

    ax.set_title(title, pad=40, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()