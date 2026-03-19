import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================================
# 1. 定义二进制结构（与处理主程序完全一致）
# =====================================================================
file_header_dtype = np.dtype([
    ('day', 'i2'), ('month', 'i2'), ('year', 'i2'), ('hour', 'i2'), ('minute', 'i2'),
    ('geo', 'S20'), ('backup0', 'i1'), ('met', 'i1'), ('backup1', '20i1'),
    ('LST', 'i1'), ('pro', 'i1'), ('kan', 'i1'), ('backup2', 'i1'),
    ('ab1', 'i2'), ('ab2', 'i2'), ('ab3', 'i2'), ('key25', 'i2'),
    ('Tok', 'f4'), ('Ndt', 'i2'), ('T0', 'i2'), ('Tgu', 'i1'), ('Ddt', 'i1'),
    ('Tpi', 'i1'), ('Ngu', 'i1'), ('colibr_period', 'i4'), ('colibr_pulse', 'i4'),
    ('backup3', '8i1'), ('Nom', 'i2'), ('kdt', 'i2'), ('Pr2', 'i2'), ('Pima', 'i2'),
    ('Fam', 'S20'), ('Lps', 'i4'), ('backup4', 'S6'), ('backup5', 'S1'),
    ('Npb', 'i1'), ('Izm', 'i2'), ('Ntk', 'i2'), ('backup6', 'S2'), ('Lgr', 'i2'),
    ('Ntik', 'i2'), ('Nst', 'i2'), ('backup7', 'S6'), ('backup8', 'S1'), ('Vup', 'i1'),
    ('Com', 'S48'), ('Isw', '90i2'), ('ProgNum', 'i2'), ('fNid', 'i4'),
    ('Pd_16', 'i2'), ('Pd_24', 'i4'), ('StartMode', 'i2'), ('StartExt', 'i2'),
    ('Pac', 'i2'), ('T_gps', 'i4'), ('T_kp', 'i4'), ('backup9', 'S18'),
    ('iADType', 'i2'), ('backup10', 'S86')
])

chan_header_dtype = np.dtype([
    ('Idk', 'i1'), ('standby0', 'i1'), ('Uko', 'i1'), ('Ufl', 'i1'),
    ('Pkt', 'i2'), ('Prf', 'i2'), ('Damp', 'i2'), ('Ddac', 'i2'),
    ('standby3', 'S2'), ('Razm', 'i2'), ('Nvt', 'i2'), ('Ubal', 'i2'),
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('Ecs', 'f4'), ('standby5', 'S4'),
    ('k1', 'f4'), ('k10', 'f4'), ('k100', 'f4'), ('k1000', 'f4'),
    ('Ugl', 'f4'), ('Er', 'f4')
])


def read_age_file(filepath):
    is_tx = 'st01' in Path(filepath).stem.lower()
    with open(filepath, 'rb') as f:
        file_header = np.frombuffer(f.read(512), dtype=file_header_dtype)[0]
        n_chan = int(file_header['kan'])
        chan_headers = np.frombuffer(f.read(64 * n_chan), dtype=chan_header_dtype)

        if file_header['Vup'] < 100:
            sample_rate = 1000.0 / (2.0 ** (file_header['pro'] - 1))
        else:
            sample_rate = float(file_header['Ndt'])
        n_max_period = file_header['Pima']

        f.seek(2048)
        raw_data = np.fromfile(f, dtype=np.int32).astype(np.float64)
        n_total_sam = len(raw_data) // n_chan
        timeseries = raw_data[:n_total_sam * n_chan].reshape((n_chan, n_total_sam), order='F')

        for i in range(n_chan):
            adc = chan_headers[i]['Ecs']
            if np.abs(adc) < 1e-15 or np.isnan(adc): adc = 1.0

            uko = chan_headers[i]['Uko']
            gain_map = {1: 'k1', 2: 'k1', 3: 'k10', 4: 'k100', 5: 'k1000'}
            gain = chan_headers[i][gain_map.get(uko, 'k1')]
            if np.abs(gain) < 1e-15 or np.isnan(gain): gain = 1.0

            timeseries[i, :] = timeseries[i, :] * (adc / gain)

            mn = 1.0
            if not is_tx:
                idk = chan_headers[i]['Idk']
                razm = chan_headers[i]['Razm']
                if idk in [2, 3]:
                    mn = float(razm)
                else:
                    mapping = {1: 200000.0, 2: 65000.0, 3: 50000.0, 4: 100000.0, 8: 25000.0}
                    mn = mapping.get(razm, 65000.0)
                if np.abs(mn) < 1e-10 or np.isnan(mn): mn = 1.0
                timeseries[i, :] = timeseries[i, :] / mn

        timeseries = np.nan_to_num(timeseries, nan=0.0, posinf=0.0, neginf=0.0)
    return file_header, chan_headers, timeseries, sample_rate, n_max_period


# =====================================================================
# 2. 波形提取与绘图功能
# =====================================================================
def plot_age_waveform(filepath, target_period=1, target_channel=0, num_cycles_to_plot=3):
    """
    filepath: 文件路径
    target_period: 想查看第几个频点（周期），从 1 开始
    target_channel: 想查看哪个通道（0代表通道1，1代表通道2）
    num_cycles_to_plot: 在图上显示多少个循环（全画出来会密密麻麻看不清，通常画 3~5 个循环就能看清波形）
    """
    print(f"正在读取文件: {filepath} ...")
    header, chans, ts, sr, periods = read_age_file(filepath)

    if target_period > periods:
        print(f"错误: 请求的周期数 ({target_period}) 超过了文件最大周期数 ({periods})")
        return

    # 定位目标周期在总时间序列中的起始位置
    start_idx = 0
    for i in range(target_period - 1):
        c_len = int(header['Isw'][30 + i])
        c_num = int(header['Isw'][60 + i])
        start_idx += c_len * c_num

    # 获取当前目标周期的参数
    cyc_len = int(header['Isw'][30 + target_period - 1])
    cyc_num = int(header['Isw'][60 + target_period - 1])
    n_sam = cyc_len * cyc_num

    # 切片提取出该周期该通道的完整波形
    segment = ts[target_channel, start_idx: start_idx + n_sam]

    # 截取前几个循环用于清晰展示
    points_to_plot = cyc_len * min(cyc_num, num_cycles_to_plot)
    plot_data = segment[:points_to_plot]

    # 生成时间轴 (秒)
    time_axis = np.arange(points_to_plot) / sr

    # 开始画图
    plt.figure(figsize=(14, 5))

    # 画出波形折线图（带数据点标记）
    plt.plot(time_axis, plot_data, color='#1f77b4', linestyle='-', linewidth=1.5, marker='.', markersize=4)

    # 用垂直虚线标出每一个完整循环(Cycle)的分割边界
    for i in range(1, min(cyc_num, num_cycles_to_plot) + 1):
        plt.axvline(x=(i * cyc_len) / sr, color='red', linestyle='--', alpha=0.5)

    filename = Path(filepath).name
    plt.title(f"Waveform: {filename} | Period: {target_period} | Channel: {target_channel + 1}\n"
              f"(Showing first {min(cyc_num, num_cycles_to_plot)} cycles, {cyc_len} pts/cycle)", fontsize=14)
    plt.xlabel("Time (Seconds)", fontsize=12)
    plt.ylabel("Amplitude (Calibrated)", fontsize=12)
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


# =====================================================================
# 3. 运行区：在此处修改你要对比的文件和通道
# =====================================================================
if __name__ == '__main__':
    # 【测试文件 1：发射机】
    # 查看发射机的通道 2 (电流)，对应 Python 索引 1
    tx_filepath = r"D:\资料包\时频电磁\测试数据\current\07-14\C016ST01.DAT"
    plot_age_waveform(
        filepath=tx_filepath,
        target_period=1,  # 查看第 1 周期
        target_channel=1,  # 查看通道 2 (电流)
        num_cycles_to_plot=3  # 只画出前 3 个循环，避免挤在一起
    )

    # 【测试文件 2：接收机】
    # 查看接收机的通道 1 (电场Ex)，对应 Python 索引 0
    rx_filepath = r"D:\资料包\时频电磁\测试数据\data\0714\C016ST512.dat"
    plot_age_waveform(
        filepath=rx_filepath,
        target_period=1,  # 查看第 1 周期
        target_channel=0,  # 查看通道 1 (电场Ex)
        num_cycles_to_plot=3  # 只画出前 3 个循环
    )