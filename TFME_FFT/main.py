import os
import numpy as np
from pathlib import Path
from scipy.fft import fft, fftfreq
# =====================================================================
# 第一部分：二进制数据读取模块
# =====================================================================

# 定义AGE文件头的二进制数据结构（512字节）
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

# 定义AGE通道头的二进制数据结构（每个通道64字节）
chan_header_dtype = np.dtype([
    ('Idk', 'i1'), ('standby0', 'i1'), ('Uko', 'i1'), ('Ufl', 'i1'),
    ('Pkt', 'i2'), ('Prf', 'i2'), ('Damp', 'i2'), ('Ddac', 'i2'),
    ('standby3', 'S2'), ('Razm', 'i2'), ('Nvt', 'i2'), ('Ubal', 'i2'),
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('Ecs', 'f4'), ('standby5', 'S4'),
    ('k1', 'f4'), ('k10', 'f4'), ('k100', 'f4'), ('k1000', 'f4'),
    ('Ugl', 'f4'), ('Er', 'f4')
])


def check_file_is_tx(filepath):
    # 检查文件名是否包含'st01'，判断是否为发射机(Tx)文件
    return 'st01' in Path(filepath).stem.lower()


def read_age_file(filepath):
    # 判断当前文件是Tx还是Rx
    is_tx = check_file_is_tx(filepath)
    # 以二进制只读模式打开文件
    with open(filepath, 'rb') as f:
        # 读取前512字节作为文件头，并转换为结构化数组
        file_header = np.frombuffer(f.read(512), dtype=file_header_dtype)[0]
        # 从文件头获取通道总数 ('kan'参数)
        n_chan = int(file_header['kan'])
        # 紧接着读取64*n_chan字节作为所有通道的头信息
        chan_headers = np.frombuffer(f.read(64 * n_chan), dtype=chan_header_dtype)

        # 根据'Vup'参数计算采样率
        if file_header['Vup'] < 100:
            # 老版本仪器：采样率由'pro'参数决定
            sample_rate = 1000.0 / (2.0 ** (file_header['pro'] - 1))
        else:
            # 新版本仪器：'Ndt'直接表示采样率
            sample_rate = float(file_header['Ndt'])
        # 获取最大周期数 ('Pima'参数)
        n_max_period = file_header['Pima']

        # 跳过文件头填充区，直接跳到第2048字节开始读取实际数据
        f.seek(2048)
        # 读取原始数据：AGE底层存储为int32，读取后转为float64以便后续计算
        raw_data = np.fromfile(f, dtype=np.int32).astype(np.float64)
        # 计算总采样点数：总数据量 / 通道数
        n_total_sam = len(raw_data) // n_chan
        # 将一维数组重塑为二维数组 (通道数, 采样点数)，order='F'表示Fortran列优先顺序
        timeseries = raw_data[:n_total_sam * n_chan].reshape((n_chan, n_total_sam), order='F')

        # 初始化增益列表和归一化系数列表
        gains = []
        mn_list = []
        # 遍历每个通道进行物理量还原
        for i in range(n_chan):
            # 获取ADC满量程系数 ('Ecs')
            adc = chan_headers[i]['Ecs']
            # 保护：如果adc为0或NaN，强制设为1.0避免除零
            if np.abs(adc) < 1e-15 or np.isnan(adc): adc = 1.0

            # 获取档位码 ('Uko')
            uko = chan_headers[i]['Uko']
            # 根据档位码映射到对应的增益系数 (k1, k10, k100, k1000)
            gain_map = {1: 'k1', 2: 'k1', 3: 'k10', 4: 'k100', 5: 'k1000'}
            gain = chan_headers[i][gain_map.get(uko, 'k1')]
            # 保护：如果增益为0或NaN，强制设为1.0
            if np.abs(gain) < 1e-15 or np.isnan(gain): gain = 1.0
            gains.append(gain)

            # 物理量还原：原始值 * (ADC系数 / 增益)
            timeseries[i, :] = timeseries[i, :] * (adc / gain)

            # 初始化归一化系数mn
            mn = 1.0
            # 仅对接收机(Rx)数据进行进一步归一化，发射机(Tx)不需要
            if not is_tx:
                # 获取通道类型 ('Idk')
                idk = chan_headers[i]['Idk']
                # 获取量程 ('Razm')
                razm = chan_headers[i]['Razm']
                # 如果是磁场通道(Idk=2或3)，mn直接取razm
                if idk in [2, 3]:
                    mn = float(razm)
                else:
                    # 如果是电场通道，根据razm映射到固定的电极距系数
                    mapping = {1: 200000.0, 2: 65000.0, 3: 50000.0, 4: 100000.0, 8: 25000.0}
                    mn = mapping.get(razm, 65000.0)
                # 保护：如果mn为0或NaN，强制设为1.0
                if np.abs(mn) < 1e-10 or np.isnan(mn): mn = 1.0
                # 除以mn，归一化到单位装置尺寸
                timeseries[i, :] = timeseries[i, :] / mn
            mn_list.append(mn)

        # 最后清洗数据：将NaN、正无穷、负无穷全部替换为0.0
        timeseries = np.nan_to_num(timeseries, nan=0.0, posinf=0.0, neginf=0.0)

    # 返回所有读取和处理后的数据
    return file_header, chan_headers, timeseries, sample_rate, n_max_period, gains, mn_list


# =====================================================================
# 第二部分：电磁场正演与反演计算模块
# =====================================================================
def EXY(ns, FP, CUR, sigma, sx1, sx2, sy1, sy2, x0, y0):
    # 步骤1：计算电磁学基础参数
    OMG = 2.0 * np.pi * FP  # 计算角频率 ω=2πf
    mu = 4.0 * np.pi * 1.0e-7  # 真空磁导率 (H/m)
    YBCL = 8.854187817e-12  # 真空介电常数 (F/m)
    Mz = 1j * mu * OMG  # 复磁导率（引入虚数单位j）
    my = sigma + 1j * YBCL * OMG  # 复电导率（σ + jωε）
    k1 = np.sqrt(-Mz * my)  # 计算波数 k

    # 步骤2：离散发射源（将有限长导线拆分为ns个电偶极子）
    ds = abs(sx2 - sx1) / ns  # 计算每个小段的长度
    # 计算每个小段的中点坐标
    zd = min(sx1, sx2) + np.arange(ns) * ds + ds / 2.0
    # 计算每个小段中点到接收点的距离
    rxy = np.sqrt((x0 - zd) ** 2 + (y0 - sy1) ** 2)
    # 计算夹角的余弦值
    cosf = np.abs(x0 - zd) / rxy
    # 计算复波数与距离的乘积 ikr
    ikr = 1j * k1 * rxy

    # 步骤3：计算每个小段的电场贡献并求和
    PE = CUR * ds / (2.0 * np.pi * rxy ** 3)  # 电场幅值系数
    # 均匀半空间水平电偶极子电场公式，求和得到总电场
    EX_complex = np.sum(PE * (3.0 * cosf ** 2 - 2.0 + np.exp(-ikr) * (1.0 + ikr)) / sigma)
    return np.abs(EX_complex)  # 返回电场幅值（取模）


def simulated_annealing(fp, exc1_target, sx1, sx2, sy1, sy2, x0, y0, ns=30, cur=1.0):
    # 初始化：设定初始猜测电导率 (0.01 S/m，对应100 Ω·m)
    sigma_current = 0.01
    # 设定初始温度 (经验值50)
    wendu0 = 50.0
    # 计算初始电导率对应的正演电场
    exs1_current = EXY(ns, fp, cur, sigma_current, sx1, sx2, sy1, sy2, x0, y0)
    # 计算初始的相对误差平方
    deta2_best = ((exs1_current - exc1_target) / exs1_current) ** 2

    # 外层循环：温度迭代（最多5000次）
    for kkk in range(1, 5001):
        # 指数降温：当前温度 = 初始温度 * 0.99^迭代次数
        wendu_current = wendu0 * (0.99 ** kkk)
        # 初始化当前温度下的成功接受次数
        ukk = 1
        # 内层循环：每个温度下的随机扰动（最多500次）
        for uuu in range(1, 501):
            # 保存当前电导率作为“旧值”
            sigma_old = sigma_current
            # 生成一个0到1之间的均匀随机数
            ux = np.random.rand()
            # 核心扰动公式：温度越高，扰动范围越大
            uux = wendu_current * np.sign(ux - 0.5) * ((1.0 + 1.0 / wendu_current) ** np.abs(2.0 * ux - 1.0) - 1.0)
            # 在视电阻率(1/σ)空间进行更新，生成新电导率
            sigma_new = 1.0 / (1.0 / sigma_old + uux * 200.0)

            # 约束条件1：视电阻率不能小于1 Ω·m
            if 1.0 / sigma_new <= 1.0: sigma_new = 1.0
            # 约束条件2：视电阻率不能大于110000 Ω·m
            if 1.0 / sigma_new >= 110000.0: sigma_new = 1.0 / 110000.0

            # 调用正演函数，计算新电导率对应的理论电场
            exs1_new = EXY(ns, fp, cur, sigma_new, sx1, sx2, sy1, sy2, x0, y0)
            # 计算新电导率的相对误差平方
            deta2_new = ((exs1_new - exc1_target) / exs1_new) ** 2

            # 计算对数误差差（避免数量级影响），并处理除零情况
            deta = 0.0 if deta2_new == 0 or deta2_best == 0 else np.log10(deta2_new) - np.log10(deta2_best)

            # Metropolis准则判断
            if deta <= 0.0:
                # 情况1：新误差更小（deta<=0），直接接受新解
                deta2_best = deta2_new  # 更新最优误差
                sigma_current = sigma_new  # 更新当前电导率
                ukk += 1  # 成功接受次数+1
            else:
                # 情况2：新误差更大（deta>0），按概率接受
                # 计算接受概率的基础值
                base = 1.0 - (1.0 - 0.25) * deta / wendu_current
                # 概率判断：如果base有效且概率大于随机数，则接受
                if base > 0 and (base ** (1.0 / (1.0 - 0.25))) >= np.random.rand():
                    deta2_best = deta2_new  # 更新最优误差
                    sigma_current = sigma_new  # 更新当前电导率
                    ukk += 1  # 成功接受次数+1
                else:
                    # 拒绝新解，恢复到旧电导率
                    sigma_current = sigma_old
            # 内层循环提前退出：如果本温度下成功接受了8次，说明探索充分
            if ukk >= 8: break
        # 外层循环提前退出：如果拟合误差<=0.001%，说明精度达标
        if np.sqrt(deta2_best) * 100.0 <= 1.0e-3: break
    # 返回最终结果：视电阻率(1/σ) 和 拟合误差(%)
    return 1.0 / sigma_current, np.sqrt(deta2_best) * 100


# =====================================================================
# 第三部分：核心主程序
# =====================================================================
def main():
    # === 请修改为你的实际路径 ===
    tx_filepath = r"D:\资料包\时频电磁\测试数据\current\07-14\C016ST01.DAT"
    rx_filepath = r"D:\资料包\时频电磁\测试数据\data\0714\C016ST512.dat"

    print(">>> [1/4] 正在加载并处理数据...")
    # 读取发射机(Tx)文件
    tx_header, _, tx_ts, tx_sr, tx_periods, _, _ = read_age_file(tx_filepath)
    # 读取接收机(Rx)文件
    rx_header, rx_chans, rx_ts, rx_sr, rx_periods, rx_gains, rx_mns = read_age_file(rx_filepath)

    # 取Tx和Rx中较小的周期数作为最大处理周期数
    n_max_period = min(tx_periods, rx_periods)

    # 铁律2：严格按照 Fortran 的通道索引对齐！
    # 接收机选第 0 个通道 (通常是Ex电场)
    chan_idx_rx = 0
    # 发射机选最后一个通道 (通常第 1 个通道才是电流 I，需根据实际情况确认)
    chan_idx_tx = int(tx_header['kan']) - 1

    print(f"  [√] 已强制锁定 RX 通道: {chan_idx_rx} (电场)")
    print(f"  [√] 已强制锁定 TX 通道: {chan_idx_tx} (电流)\n")

    # 初始化数组：频率、发射电流复数、接收电场复数
    fre = np.zeros(n_max_period)
    CURc = np.zeros(n_max_period, dtype=complex)
    Exc = np.zeros(n_max_period, dtype=complex)

    print(">>> [2/4] 正在进行 FFT 提取特征波形...")
    # 初始化Tx和Rx的数据切片索引
    curr_tx_idx = 0
    curr_rx_idx = 0

    # 遍历每个频率周期进行处理
    for i in range(n_max_period):
        # 从文件头'Isw'参数读取：第30+i个是单周期长度
        cyc_len = int(rx_header['Isw'][30 + i])#30 ~ 59：第 i 个频率周期的 “单周期采样点数” 存在 Isw[30+i]
        # 从文件头'Isw'参数读取：第60+i个是叠加次数
        cyc_num = int(rx_header['Isw'][60 + i])#60 ~ 89：第 i 个频率周期的 “叠加次数” 存在 Isw[60+i]
        # 计算本周期的总采样点数
        n_sam = cyc_len * cyc_num

        # 保护：如果采样点数<=0，跳过本周期
        if n_sam <= 0: continue
        # 保护：如果索引越界，停止处理
        if curr_rx_idx + n_sam > rx_ts.shape[1] or curr_tx_idx + n_sam > tx_ts.shape[1]: break

        # 计算本周期的基频 f0 = 1 / (周期长度 / 采样率)
        freq0 = 1.0 / (cyc_len / rx_sr)
        fre[i] = freq0

        # 切片提取本周期的Tx和Rx数据
        tx_seg = tx_ts[chan_idx_tx, curr_tx_idx: curr_tx_idx + n_sam]
        rx_seg = rx_ts[chan_idx_rx, curr_rx_idx: curr_rx_idx + n_sam]

        # --- 对Rx数据做FFT ---
        yf_rx = fft(rx_seg)
        xf_rx = fftfreq(len(rx_seg), 1.0 / rx_sr)
        # 找到频率轴上最接近基频f0的索引
        idx_rx = np.argmin(np.abs(xf_rx - freq0))
        # 提取该频率点的复数幅值，并做归一化（*2是因为FFT是双边谱）
        Exc[i] = yf_rx[idx_rx] / len(rx_seg) * 2.0

        # --- 对Tx数据做FFT ---
        yf_tx = fft(tx_seg)
        xf_tx = fftfreq(len(tx_seg), 1.0 / tx_sr)
        # 找到频率轴上最接近基频f0的索引
        idx_tx = np.argmin(np.abs(xf_tx - freq0))
        # 提取该频率点的复数幅值
        CURc[i] = yf_tx[idx_tx] / len(tx_seg) * 2.0

        # 更新索引，移动到下一个周期的数据起点
        curr_rx_idx += n_sam
        curr_tx_idx += n_sam

    print(">>> [3/4] 准备计算广域视电阻率并反演...")

    #  Fortran 的隐藏硬件缩放
    # -1 代表修复发射机电流记录的硬件反极性（纠正 180 度相位差）
    # 1e6 代表将接收机的电场从 V 降维回 uV (微伏) 数量级
    FORTRAN_CALIBRATION = -1000000.0

    # 应用校准系数到发射电流
    CURc_calibrated = CURc * FORTRAN_CALIBRATION

    # 保护：避免电流为0导致除零错误
    CURc_safe = np.where(np.abs(CURc_calibrated) < 1e-15, 1e-15 + 1e-15j, CURc_calibrated)
    # 计算传递函数：电场 / 电流
    with np.errstate(divide='ignore', invalid='ignore'):
        transfer_func = Exc / CURc_safe
        # 将NaN和Inf替换为0
        transfer_func[np.isnan(transfer_func) | np.isinf(transfer_func)] = 0.0j

    # 提取传递函数的幅值（广域视电阻率的基础）
    xyhs2 = np.abs(transfer_func)
    # 提取传递函数的相位（单位：度）
    phase = np.degrees(np.arctan2(np.imag(transfer_func), np.real(transfer_func)))

    # === 输入观测系统坐标（示例坐标，请根据实际修改） ===
    SX1, SY1 = 4879201.50, 15405615.90  # 发射源端点1
    Sx2, Sy2 = 4890565.00, 15405562.80  # 发射源端点2
    x0, y0 = 4880126.20, 15416002.66  # 接收点坐标
    mn = 1.0  # 归一化系数

    # === 坐标旋转（将斜交的发射线旋转为水平，方便正演计算） ===
    SCD = np.sqrt((SX1 - Sx2) ** 2 + (SY1 - Sy2) ** 2)  # 计算发射线长度
    SITA = np.arcsin(np.abs(SY1 - Sy2) / SCD)  # 计算旋转角
    # 对发射源端点1进行旋转
    sx10, sy10 = SX1 * np.cos(SITA) - SY1 * np.sin(SITA), SX1 * np.sin(SITA) + SY1 * np.cos(SITA)
    # 对发射源端点2进行旋转
    sx20, sy20 = Sx2 * np.cos(SITA) - Sy2 * np.sin(SITA), Sx2 * np.sin(SITA) + Sy2 * np.cos(SITA)
    # 对接收点进行旋转
    x00, y00 = x0 * np.cos(SITA) - y0 * np.sin(SITA), x0 * np.sin(SITA) + y0 * np.cos(SITA)

    # 最终的归一化实测电场幅值（反演的目标值）
    EXC1 = xyhs2 / mn

    # 打印结果表头
    print("-" * 65)
    print(f"{'频率 (Hz)':<10} | {'视电阻率 (Ohm-m)':<20} | {'相位 (度)':<10} | {'拟合误差'}")
    print("-" * 65)

    # 遍历每个频率，调用模拟退火进行反演
    for i in range(n_max_period):
        # 保护：如果频率为0或实测电场为0，跳过
        if fre[i] == 0.0 or EXC1[i] == 0.0: continue

        # 调用模拟退火反演函数
        res_val, err_val = simulated_annealing(fre[i], EXC1[i], sx10, sx20, sy10, sy20, x00, y00, ns=30, cur=1.0)
        # 打印结果：频率、视电阻率、相位、拟合误差
        print(f"{fre[i]:<10.4f} | {res_val:<20.2f} | {phase[i]:<10.2f} | {err_val:.4f}%")


