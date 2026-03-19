import os
import numpy as np
from pathlib import Path
from scipy.fft import fft, fftfreq

# =====================================================================
# 第一部分：二进制数据读取模块
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


def check_file_is_tx(filepath):
    return 'st01' in Path(filepath).stem.lower()


def read_age_file(filepath):
    is_tx = check_file_is_tx(filepath)
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

        gains = []
        mn_list = []
        for i in range(n_chan):
            adc = chan_headers[i]['Ecs']
            if np.abs(adc) < 1e-15 or np.isnan(adc): adc = 1.0

            uko = chan_headers[i]['Uko']
            gain_map = {1: 'k1', 2: 'k1', 3: 'k10', 4: 'k100', 5: 'k1000'}
            gain = chan_headers[i][gain_map.get(uko, 'k1')]
            if np.abs(gain) < 1e-15 or np.isnan(gain): gain = 1.0
            gains.append(gain)

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
            mn_list.append(mn)

        timeseries = np.nan_to_num(timeseries, nan=0.0, posinf=0.0, neginf=0.0)

    return file_header, chan_headers, timeseries, sample_rate, n_max_period, gains, mn_list


# =====================================================================
# 第二部分：电磁场正演与反演计算模块
# =====================================================================
def EXY(ns, FP, CUR, sigma, sx1, sx2, sy1, sy2, x0, y0):
    OMG = 2.0 * np.pi * FP
    mu = 4.0 * np.pi * 1.0e-7
    YBCL = 8.854187817e-12
    Mz = 1j * mu * OMG
    my = sigma + 1j * YBCL * OMG
    k1 = np.sqrt(-Mz * my)

    ds = abs(sx2 - sx1) / ns
    zd = min(sx1, sx2) + np.arange(ns) * ds + ds / 2.0
    rxy = np.sqrt((x0 - zd) ** 2 + (y0 - sy1) ** 2)
    cosf = np.abs(x0 - zd) / rxy
    ikr = 1j * k1 * rxy

    PE = CUR * ds / (2.0 * np.pi * rxy ** 3)
    EX_complex = np.sum(PE * (3.0 * cosf ** 2 - 2.0 + np.exp(-ikr) * (1.0 + ikr)) / sigma)
    return np.abs(EX_complex)


def simulated_annealing(fp, exc1_target, sx1, sx2, sy1, sy2, x0, y0, ns=30, cur=1.0):
    sigma_current = 0.01
    wendu0 = 50.0
    exs1_current = EXY(ns, fp, cur, sigma_current, sx1, sx2, sy1, sy2, x0, y0)
    deta2_best = ((exs1_current - exc1_target) / exs1_current) ** 2

    for kkk in range(1, 5001):
        wendu_current = wendu0 * (0.99 ** kkk)
        ukk = 1
        for uuu in range(1, 501):
            sigma_old = sigma_current
            ux = np.random.rand()
            uux = wendu_current * np.sign(ux - 0.5) * ((1.0 + 1.0 / wendu_current) ** np.abs(2.0 * ux - 1.0) - 1.0)
            sigma_new = 1.0 / (1.0 / sigma_old + uux * 200.0)

            if 1.0 / sigma_new <= 1.0: sigma_new = 1.0
            if 1.0 / sigma_new >= 110000.0: sigma_new = 1.0 / 110000.0

            exs1_new = EXY(ns, fp, cur, sigma_new, sx1, sx2, sy1, sy2, x0, y0)
            deta2_new = ((exs1_new - exc1_target) / exs1_new) ** 2

            deta = 0.0 if deta2_new == 0 or deta2_best == 0 else np.log10(deta2_new) - np.log10(deta2_best)

            if deta <= 0.0:
                deta2_best = deta2_new
                sigma_current = sigma_new
                ukk += 1
            else:
                base = 1.0 - (1.0 - 0.25) * deta / wendu_current
                if base > 0 and (base ** (1.0 / (1.0 - 0.25))) >= np.random.rand():
                    deta2_best = deta2_new
                    sigma_current = sigma_new
                    ukk += 1
                else:
                    sigma_current = sigma_old
            if ukk >= 8: break
        if np.sqrt(deta2_best) * 100.0 <= 1.0e-3: break
    return 1.0 / sigma_current, np.sqrt(deta2_best) * 100


# =====================================================================
# 第三部分：辅助文件导出模块 (完全复刻 Fortran 输出功能)
# =====================================================================
def export_info_file(filepath, out_dir, header, chans, sr, periods, gains, mns):
    stem = Path(filepath).stem
    info_file = out_dir / f"{stem}.info"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"  SampleRate :    {sr:.3E}\n")
        f.write(f"  #MaxPeriod :      {periods}\n")
        f.write("         #Chan          Gain            MN           ADC           PRF           PKT           IDK\n")
        for i in range(len(chans)):
            f.write(f"  {i + 1:10}  {gains[i]:12.5f}  {mns[i]:12.5f}  {chans[i]['Ecs']:12.5f}  "
                    f"{chans[i]['Prf']:12}  {chans[i]['Pkt']:12}  {chans[i]['Idk']:12}\n")
        f.write("\n   #Period   #CycLen   #CycNum\n")
        for i in range(periods):
            cyc_len = int(header['Isw'][30 + i])
            cyc_num = int(header['Isw'][60 + i])
            f.write(f"  {i + 1:8}  {cyc_len:8}  {cyc_num:8}\n")


def export_timeseries(filepath, out_dir, period_idx, cyc_len, cyc_num, n_chan, ts_data):
    stem = Path(filepath).stem
    ts_file = out_dir / f"{stem}_#Period={period_idx:02d}_Timeseries.txt"
    with open(ts_file, 'w', encoding='utf-8') as f:
        f.write("   #Period   #CycLen   #CycNum     #Chan\n")
        f.write(f"  {period_idx:8d}  {cyc_len:8d}  {cyc_num:8d}  {n_chan:8d}\n")
        np.savetxt(f, ts_data.T, fmt="  %30.12E")


def export_freqseries(filepath, out_dir, chans, periods, freqs, fr_out, fi_out, n_chan):
    # Fortran 对应的列头映射表
    string0 = ['#realBZ', '#realEX', '#realEY', '#realHX', '#realHY', '#realHZ', '#realU', 'readI',
               '#imagBZ', '#imagEX', '#imagEY', '#imagHX', '#imagHY', '#imagHZ', '#imagU', 'imagI']
    stem = Path(filepath).stem
    fs_file = out_dir / f"{stem}_#Period=01-{periods:02d}_Freqseries.txt"

    # 生成动态表头
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


# =====================================================================
# 第四部分：核心主程序
# =====================================================================
def main():
    # === 请修改为你的实际路径 ===
    tx_filepath = r"D:\资料包\时频电磁\测试数据\current\07-14\C016ST01.DAT"
    rx_filepath = r"D:\资料包\时频电磁\测试数据\data\0714\C016ST512.dat"

    print(">>> [1/5] 正在加载并处理数据...")
    tx_header, tx_chans, tx_ts, tx_sr, tx_periods, tx_gains, tx_mns = read_age_file(tx_filepath)
    rx_header, rx_chans, rx_ts, rx_sr, rx_periods, rx_gains, rx_mns = read_age_file(rx_filepath)

    n_max_period = min(tx_periods, rx_periods)
    n_chan_tx = int(tx_header['kan'])
    n_chan_rx = int(rx_header['kan'])

    # 建立与 Fortran 相同的目录结构
    tx_path, rx_path = Path(tx_filepath), Path(rx_filepath)
    tx_ts_dir = tx_path.parent / f"{tx_path.stem}_Timeseries"
    tx_fs_dir = tx_path.parent / f"{tx_path.stem}_Freqseries"
    rx_ts_dir = rx_path.parent / f"{rx_path.stem}_Timeseries"
    rx_fs_dir = rx_path.parent / f"{rx_path.stem}_Freqseries"

    for d in [tx_ts_dir, tx_fs_dir, rx_ts_dir, rx_fs_dir]:
        d.mkdir(exist_ok=True)

    print(">>> [2/5] 正在导出 .info 描述文件...")
    export_info_file(tx_filepath, tx_ts_dir, tx_header, tx_chans, tx_sr, tx_periods, tx_gains, tx_mns)
    export_info_file(rx_filepath, rx_ts_dir, rx_header, rx_chans, rx_sr, rx_periods, rx_gains, rx_mns)

    # 严格按照 Fortran 的通道索引对齐
    chan_idx_rx = 0
    chan_idx_tx = n_chan_tx - 1
    print(f"  [√] 已强制锁定 RX 通道: {chan_idx_rx} (电场)")
    print(f"  [√] 已强制锁定 TX 通道: {chan_idx_tx} (电流)\n")

    fre = np.zeros(n_max_period)
    tx_Fr_out = np.zeros((n_max_period, n_chan_tx))
    tx_Fi_out = np.zeros((n_max_period, n_chan_tx))
    rx_Fr_out = np.zeros((n_max_period, n_chan_rx))
    rx_Fi_out = np.zeros((n_max_period, n_chan_rx))

    print(">>> [3/5] 正在进行 FFT 并导出 _Timeseries.txt 文件...")
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

        # 截取此时段数据并导出时域波形文件
        tx_seg_all = tx_ts[:, curr_tx_idx: curr_tx_idx + n_sam]
        rx_seg_all = rx_ts[:, curr_rx_idx: curr_rx_idx + n_sam]

        export_timeseries(tx_filepath, tx_ts_dir, i + 1, cyc_len, cyc_num, n_chan_tx, tx_seg_all)
        export_timeseries(rx_filepath, rx_ts_dir, i + 1, cyc_len, cyc_num, n_chan_rx, rx_seg_all)

        # 发射机各通道 FFT 提取
        for c in range(n_chan_tx):
            yf = fft(tx_seg_all[c, :])
            xf = fftfreq(n_sam, 1.0 / tx_sr)
            idx = np.argmin(np.abs(xf - freq0))
            tx_Fr_out[i, c] = np.real(yf[idx] / n_sam * 2.0)
            tx_Fi_out[i, c] = np.imag(yf[idx] / n_sam * 2.0)

        # 接收机各通道 FFT 提取
        for c in range(n_chan_rx):
            yf = fft(rx_seg_all[c, :])
            xf = fftfreq(n_sam, 1.0 / rx_sr)
            idx = np.argmin(np.abs(xf - freq0))
            rx_Fr_out[i, c] = np.real(yf[idx] / n_sam * 2.0)
            rx_Fi_out[i, c] = np.imag(yf[idx] / n_sam * 2.0)

        curr_rx_idx += n_sam
        curr_tx_idx += n_sam

    print(">>> [4/5] 正在导出 _Freqseries.txt 频域汇总文件...")
    export_freqseries(tx_filepath, tx_fs_dir, tx_chans, n_max_period, fre, tx_Fr_out, tx_Fi_out, n_chan_tx)
    export_freqseries(rx_filepath, rx_fs_dir, rx_chans, n_max_period, fre, rx_Fr_out, rx_Fi_out, n_chan_rx)

    print(">>> [5/5] 准备计算广域视电阻率并反演...")
    # 直接由复数矩阵提取目标计算通道
    Exc = rx_Fr_out[:, chan_idx_rx] + 1j * rx_Fi_out[:, chan_idx_rx]
    CURc = tx_Fr_out[:, chan_idx_tx] + 1j * tx_Fi_out[:, chan_idx_tx]

    # 【标定恢复】：-1修复反极性，1e6把电场还原为微伏级计算
    FORTRAN_CALIBRATION = -1000000.0
    CURc_calibrated = CURc * FORTRAN_CALIBRATION

    CURc_safe = np.where(np.abs(CURc_calibrated) < 1e-15, 1e-15 + 1e-15j, CURc_calibrated)
    with np.errstate(divide='ignore', invalid='ignore'):
        transfer_func = Exc / CURc_safe
        transfer_func[np.isnan(transfer_func) | np.isinf(transfer_func)] = 0.0j

    xyhs2 = np.abs(transfer_func)
    phase = np.degrees(np.arctan2(np.imag(transfer_func), np.real(transfer_func)))

    SX1, SY1 = 4879201.50, 15405615.90
    Sx2, Sy2 = 4890565.00, 15405562.80
    x0, y0 = 4880126.20, 15416002.66
    mn = 1.0
    SCD = np.sqrt((SX1 - Sx2) ** 2 + (SY1 - Sy2) ** 2)
    SITA = np.arcsin(np.abs(SY1 - Sy2) / SCD)
    sx10, sy10 = SX1 * np.cos(SITA) - SY1 * np.sin(SITA), SX1 * np.sin(SITA) + SY1 * np.cos(SITA)
    sx20, sy20 = Sx2 * np.cos(SITA) - Sy2 * np.sin(SITA), Sx2 * np.sin(SITA) + Sy2 * np.cos(SITA)
    x00, y00 = x0 * np.cos(SITA) - y0 * np.sin(SITA), x0 * np.sin(SITA) + y0 * np.cos(SITA)

    EXC1 = xyhs2 / mn

    print("-" * 65)
    print(f"{'频率 (Hz)':<10} | {'视电阻率 (Ohm-m)':<20} | {'相位 (度)':<10} | {'拟合误差'}")
    print("-" * 65)

    for i in range(n_max_period):
        if fre[i] == 0.0 or EXC1[i] == 0.0: continue

        res_val, err_val = simulated_annealing(fre[i], EXC1[i], sx10, sx20, sy10, sy20, x00, y00, ns=30, cur=1.0)
        print(f"{fre[i]:<10.4f} | {res_val:<20.2f} | {phase[i]:<10.2f} | {err_val:.4f}%")


if __name__ == '__main__':
    main()