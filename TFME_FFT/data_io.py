import numpy as np
from pathlib import Path

# =====================================================================
# 1. 定义二进制结构字典 (AGE 仪器格式标准)
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


# =====================================================================
# 2. 核心读取函数 (提取纯原始数据)
# =====================================================================
def read_age_binary(filepath):
    """
    读取 AGE 二进制文件，直接提取原始数据机器码，不进行 ADC 和 MN 等物理量转换。
    返回: file_header, chan_headers, timeseries_matrix, sample_rate, max_periods
    """
    is_tx = 'st01' in Path(filepath).stem.lower()

    with open(filepath, 'rb') as f:
        # 1. 读取 512 字节的文件头
        file_header = np.frombuffer(f.read(512), dtype=file_header_dtype)[0]
        n_chan = int(file_header['kan'])

        # 2. 读取 64 * n_chan 字节的通道头
        chan_headers = np.frombuffer(f.read(64 * n_chan), dtype=chan_header_dtype)

        # 3. 计算真实采样率
        if file_header['Vup'] < 100:
            sample_rate = 1000.0 / (2.0 ** (file_header['pro'] - 1))
        else:
            sample_rate = float(file_header['Ndt'])

        n_max_period = file_header['Pima']

        # 4. 跳到 2048 字节处读取波形数据区
        f.seek(2048)
        raw_data = np.fromfile(f, dtype=np.int32).astype(np.float64)
        n_total_sam = len(raw_data) // n_chan

        # Fortran 按列存储，所以用 order='F'
        timeseries = raw_data[:n_total_sam * n_chan].reshape((n_chan, n_total_sam), order='F')

        # 清理异常值
        timeseries = np.nan_to_num(timeseries, nan=0.0, posinf=0.0, neginf=0.0)

    return file_header, chan_headers, timeseries, sample_rate, n_max_period


# =====================================================================
# 3. 辅助文件导出函数
# =====================================================================
def export_info_file(filepath, out_dir, header, chans, sr, periods):
    """导出与 Fortran 一致的 .info 硬件描述文件"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filepath).stem
    info_file = out_dir / f"{stem}.info"

    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"  SampleRate :    {sr:.3E}\n")
        f.write(f"  #MaxPeriod :      {periods}\n")
        f.write("         #Chan          Gain            MN           ADC           PRF           PKT           IDK\n")

        for i in range(len(chans)):
            uko = chans[i]['Uko']
            gain_map = {1: 'k1', 2: 'k1', 3: 'k10', 4: 'k100', 5: 'k1000'}
            gain = chans[i][gain_map.get(uko, 'k1')]
            if np.abs(gain) < 1e-15 or np.isnan(gain): gain = 1.0

            mn = 1.0
            f.write(f"  {i + 1:10}  {gain:12.5f}  {mn:12.5f}  {chans[i]['Ecs']:12.5f}  "
                    f"{chans[i]['Prf']:12}  {chans[i]['Pkt']:12}  {chans[i]['Idk']:12}\n")

        f.write("\n   #Period   #CycLen   #CycNum\n")
        for i in range(periods):
            cyc_len = int(header['Isw'][30 + i])
            cyc_num = int(header['Isw'][60 + i])
            f.write(f"  {i + 1:8}  {cyc_len:8}  {cyc_num:8}\n")


def export_timeseries(filepath, out_dir, period_idx, cyc_len, cyc_num, n_chan, ts_data):
    """导出单周期时域波形文件"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filepath).stem
    ts_file = out_dir / f"{stem}_#Period={period_idx:02d}_Timeseries.txt"

    with open(ts_file, 'w', encoding='utf-8') as f:
        f.write("   #Period   #CycLen   #CycNum     #Chan\n")
        f.write(f"  {period_idx:8d}  {cyc_len:8d}  {cyc_num:8d}  {n_chan:8d}\n")
        np.savetxt(f, ts_data.T, fmt="  %30.12E")


def export_freqseries(filepath, out_dir, chans, periods, freqs, fr_out, fi_out, n_chan):
    """导出频域目标频点汇总结果表"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    string0 = ['#realBZ', '#realEX', '#realEY', '#realHX', '#realHY', '#realHZ', '#realU', 'readI',
               '#imagBZ', '#imagEX', '#imagEY', '#imagHX', '#imagHY', '#imagHZ', '#imagU', 'imagI']
    stem = Path(filepath).stem
    fs_file = out_dir / f"{stem}_#Period=01-{periods:02d}_Freqseries.txt"

    headers_str = f"  {'#Period':<8}  {'#freqs':<14}"
    for i in range(n_chan):
        idk = chans[i]['Idk']
        if 0 < idk <= 8:
            real_s = string0[idk - 1]
            imag_s = string0[idk - 1 + 8]
        else:
            real_s = "#real-"
            imag_s = "#imag-"
        headers_str += f"{real_s:>20}  {imag_s:>20}  "

    with open(fs_file, 'w', encoding='utf-8') as f:
        f.write(headers_str + "\n")
        for i in range(periods):
            line = f"  {i + 1:8d}  {freqs[i]:20.5e}"
            for c in range(n_chan):
                line += f"  {fr_out[i, c]:20.8e}  {fi_out[i, c]:20.8e}"
            f.write(line + "\n")


def process_and_export_all_data(tx_filepath, rx_filepath):
    """
    一键执行：读取二进制文件、建立目录、导出 info 文件、
    按周期切片导出时域波形，并计算全频段 FFT 导出频域数据。
    """
    print(">>> [1/3] 正在加载并处理数据...")
    # 注意：在 data_io.py 内部调用自己的函数，不需要加 data_io. 前缀
    tx_header, tx_chans, tx_ts, tx_sr, tx_periods = read_age_binary(tx_filepath)
    rx_header, rx_chans, rx_ts, rx_sr, rx_periods = read_age_binary(rx_filepath)

    n_max_period = min(tx_periods, rx_periods)
    n_chan_tx = int(tx_header['kan'])
    n_chan_rx = int(rx_header['kan'])

    # 建立输出目录
    tx_path, rx_path = Path(tx_filepath), Path(rx_filepath)
    tx_ts_dir = tx_path.parent / f"{tx_path.stem}_Timeseries"
    tx_fs_dir = tx_path.parent / f"{tx_path.stem}_FreqSeries"
    rx_ts_dir = rx_path.parent / f"{rx_path.stem}_Timeseries"
    rx_fs_dir = rx_path.parent / f"{rx_path.stem}_FreqSeries"

    for d in [tx_ts_dir, rx_ts_dir, rx_fs_dir, tx_fs_dir]:
        d.mkdir(exist_ok=True)

    print(">>> [2/3] 正在导出 .info 描述文件...")
    export_info_file(tx_filepath, tx_ts_dir, tx_header, tx_chans, tx_sr, tx_periods)
    export_info_file(rx_filepath, rx_ts_dir, rx_header, rx_chans, rx_sr, rx_periods)

    chan_idx_rx = 0
    chan_idx_tx = n_chan_tx - 1
    print(f"  [√] 已强制锁定 RX 通道: {chan_idx_rx} (电场)")
    print(f"  [√] 已强制锁定 TX 通道: {chan_idx_tx} (电流)\n")

    fre = np.zeros(n_max_period)

    print(">>> [3/3] 正在进行 FFT 并导出 _Timeseries.txt 和 _Spectrum.txt 文件...")
    curr_tx_idx = 0
    curr_rx_idx = 0

    for i in range(n_max_period):
        cyc_len = int(rx_header['Isw'][30 + i])
        cyc_num = int(rx_header['Isw'][60 + i])
        n_sam = cyc_len * cyc_num

        if n_sam <= 0: continue
        if curr_rx_idx + n_sam > rx_ts.shape[1] or curr_tx_idx + n_sam > tx_ts.shape[1]: break

        freq0 = 1.0 / (cyc_len / rx_sr)
        fre[i] = freq0

        tx_seg_all = tx_ts[:, curr_tx_idx: curr_tx_idx + n_sam]
        rx_seg_all = rx_ts[:, curr_rx_idx: curr_rx_idx + n_sam]

        # 导出时域数据 (已修复 RX 导出时的 n_chan_rx 错误)
        export_timeseries(tx_filepath, tx_ts_dir, i + 1, cyc_len, cyc_num, n_chan_tx, tx_seg_all)
        export_timeseries(rx_filepath, rx_ts_dir, i + 1, cyc_len, cyc_num, n_chan_rx, rx_seg_all)

        # ---------------- 发射机 (TX) 全频段计算与导出 ----------------
        tx_export_data = []
        xf_tx = None

        for c in range(n_chan_tx):
            tx_single_chan = tx_seg_all[c, :]
            import signal_processing  # 如果顶部没导，这里确保能用到
            xf, yf = signal_processing.fft_no_stack(tx_single_chan, tx_sr)

            if c == 0:
                xf_tx = xf
                tx_export_data.append(xf_tx)

            yf_scaled = yf / n_sam * 2.0
            tx_export_data.append(np.real(yf_scaled))
            tx_export_data.append(np.imag(yf_scaled))

        tx_header_str = "Freq(Hz)"
        for c in range(n_chan_tx):
            tx_header_str += f"\tCh{c + 1}_Re\tCh{c + 1}_Im"

        tx_out_matrix = np.column_stack(tx_export_data)
        tx_out_file = tx_fs_dir / f"{tx_path.stem}_#Period={i + 1:02d}_Spectrum.txt"
        np.savetxt(tx_out_file, tx_out_matrix, fmt="%.6e", delimiter="\t", header=tx_header_str, comments="")

        # ---------------- 接收机 (RX) 全频段计算与导出 ----------------
        rx_export_data = []
        xf_rx = None

        for c in range(n_chan_rx):
            rx_single_chan = rx_seg_all[c, :]
            xf, yf = signal_processing.fft_no_stack(rx_single_chan, rx_sr)

            if c == 0:
                xf_rx = xf
                rx_export_data.append(xf_rx)

            yf_scaled = yf / n_sam * 2.0
            rx_export_data.append(np.real(yf_scaled))
            rx_export_data.append(np.imag(yf_scaled))

        rx_header_str = "Freq(Hz)"
        for c in range(n_chan_rx):
            rx_header_str += f"\tCh{c + 1}_Re\tCh{c + 1}_Im"

        rx_out_matrix = np.column_stack(rx_export_data)
        rx_out_file = rx_fs_dir / f"{rx_path.stem}_#Period={i + 1:02d}_Spectrum.txt"
        np.savetxt(rx_out_file, rx_out_matrix, fmt="%.6e", delimiter="\t", header=rx_header_str, comments="")

        curr_tx_idx += n_sam
        curr_rx_idx += n_sam

    print("[√] 数据解析并导出完毕！文件已存入同级目录。")