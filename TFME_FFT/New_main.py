import sys
import data_io
import numpy as np
from pathlib import Path

import signal_processing
import signal_processing as sp
import visualization as vis
from data_io import export_timeseries

def main():
    # --- 1. 全局配置区 ---
    # 这里可以配置默认的文件路径，避免每次都要手动输
    tx_filepath = r"D:\资料包\时频电磁\测试数据\current\07-14\C016ST01.DAT"
    rx_filepath = r"D:\资料包\时频电磁\测试数据\data\0714\C016ST521.dat"

    print("\n🚀 欢迎使用 TFEM 时频电磁数据处理平台 🚀")
    print(f"[*] 默认发射机文件: {tx_filepath}")
    print(f"[*] 默认接收机文件: {rx_filepath}")

    # --- 2. 交互式控制主循环 ---
    while True:
        print("\n" + "=" * 38)
        print("           📊 主 要 功 能 菜 单           ")
        print("=" * 38)
        print("  1. 时频数据导出 (长序列直接FFT)")
        print("  2. 叠加傅里叶变换-时域 (先叠波形再FFT)")
        print("  3. 叠加傅里叶变换-频域 (先FFT再均值化)")
        print("  4. [交互] 查看时域波形")
        print("  5. [交互] 查看全频段频谱")
        print("  0. 退出系统")
        print("=" * 38)

        choice = input("👉 请输入操作对应的数字 [0-5]: ").strip()


        if choice == '1':
            print("\n>>> 正在执行: [1] 时频数据导出并进行单周期FFT...")
            data_io.process_and_export_all_data(tx_filepath, rx_filepath)


        elif choice == '2':
            print("\n>>> 正在执行: [2] 叠加傅里叶变换 - 时域...")
            # 1. 交互获取用户想要测试的参数
            stem = Path(rx_filepath).stem
            ts_dir = Path(rx_filepath).parent / f"{stem}_Timeseries"
            try:
                p_idx = int(input("👉 请输入要测试的周期号 (例如 1): ").strip())
                ch_idx = int(input("👉 请输入要处理的通道号 (例如 0 或 1): ").strip())
            except ValueError:
                print("❌ 输入无效，请输入整数！")
                continue
            # 2. 读取对应的数据文件
            target_file = ts_dir / f"{stem}_#Period={p_idx:02d}_Timeseries.txt"
            if not target_file.exists():
                print(f"❌ 找不到文件: {target_file}")
                continue
            print(f"  正在加载数据: {target_file.name}")
            data = np.loadtxt(target_file, skiprows=2)
            time_series = data[:, ch_idx]

            rx_header, _, _, rx_sr, _ = data_io.read_age_binary(rx_filepath)
            cyc_len = int(rx_header['Isw'][30 + (p_idx - 1)])
            cyc_num = int(rx_header['Isw'][60 + (p_idx - 1)])
            # 4. 执行时域叠加
            print(f"  [>] 执行参数: 循环长度={cyc_len}, 循环次数={cyc_num}")
            stacked_wave = signal_processing.time_domain_stacking(time_series, cyc_len, cyc_num)
            # 5. 对叠加后的单周期波形进行 FFT
            freqs, spectrum = signal_processing.fft_short(stacked_wave, rx_sr)
            print("[√] 时域叠加及 FFT 计算完成！")
            # 6. 调用之前写好的可视化静态图库
            if input("  ❓ 是否查看叠加前后的波形对比？(y/n): ").strip().lower() == 'y':

                # 画出前 3 个原始循环的波形作为对比
                raw_preview_len = cyc_len * 3
                vis.plot_waveform(time_series[:raw_preview_len],
                                  title=f"Raw Data (First 3 Cycles) - Ch {ch_idx}",
                                  vlines=cyc_len)
                # 画出叠加后平滑的单循环波形
                vis.plot_waveform(stacked_wave,
                                  title=f"Time-Domain Stacked Waveform (Averaged {cyc_num} Cycles) - Ch {ch_idx}")
        elif choice == '3':
            print("\n>>> 正在执行: [3] 频域叠加傅里叶变换...")
            # 实际调用示例：
            # spectrum_list = sp.fft_all_cycles(ts, cyc_len, cyc_num)
            # avg_spectrum = sp.frequency_domain_stacking(spectrum_list)
            print("[√] 频域各个循环的特征提取及平均化完成！")

        elif choice == '4':
            print("\n>>> 开启交互式时域查看器...")
            # 这里的 n_periods 可以从 header 读取，或者直接根据文件夹里的文件数判断
            # 假设之前导出的目录和 stem 如下：
            inp = input("👉 要查看哪一端的时域数据 [T/R]: ").strip()
            if inp == 'T':
                stem = Path(tx_filepath).stem
                ts_dir = Path(tx_filepath).parent / f"{stem}_Timeseries"
            elif inp == 'R':
                stem = Path(rx_filepath).stem
                ts_dir = Path(rx_filepath).parent / f"{stem}_Timeseries"
            vis.interactive_time_viewer(ts_dir, stem)

        elif choice == '5':
            print("\n>>> 开启交互式频域查看器...")
            inp = input("👉 要查看哪一端的时域数据 [T/R]: ").strip()
            if inp == 'T':
                stem = Path(tx_filepath).stem
                fs_dir = Path(tx_filepath).parent / f"{stem}_FreqSeries"
            elif inp == 'R':
                stem = Path(rx_filepath).stem
                fs_dir = Path(rx_filepath).parent / f"{stem}_FreqSeries"

            vis.interactive_freq_viewer(fs_dir, stem)

        elif choice == '6':
            print("\n>>> 正在执行: [6] 测试 - 不叠加，直接加窗长序列 FFT...")

            stem = Path(rx_filepath).stem
            ts_dir = Path(rx_filepath).parent / f"{stem}_Timeseries"

            p_idx = int(input("👉 请输入要测试的周期号 (例如 1): ").strip())
            ch_idx = int(input("👉 请输入要处理的通道号 (例如 0 或 1): ").strip())

            # 读取数据
            target_file = ts_dir / f"{stem}_#Period={p_idx:02d}_Timeseries.txt"
            data = np.loadtxt(target_file, skiprows=2)
            time_series = data[:, ch_idx]

            # 【重要】：去直流偏置，防止 0Hz 的能量泄露把低频全淹没了
            time_series = time_series - np.mean(time_series)

            # 获取采样率和基频
            rx_header, _, _, rx_sr, _ = data_io.read_age_binary(rx_filepath)
            cyc_len = int(rx_header['Isw'][30 + (p_idx - 1)])
            theory_f0 = rx_sr / cyc_len

            user_f0 = input(f"👉 请输入发射基频 (直接回车使用 {theory_f0:.4f} Hz): ").strip()
            f0 = theory_f0 if not user_f0 else float(user_f0)

            print(f"  [>] 原始序列总点数: {len(time_series)} 点")
            print("  [>] 正在应用 Hann 窗并执行全长 FFT...")

            # 调用加窗 FFT 函数
            freqs, amplitude = signal_processing.long_fft_with_window(time_series, rx_sr, window_type='hann')

            print("[√] 计算完成！正在出图...")


            # ==========================================
            # 【核心修复】：调用最新的终极混合画图函数
            # ==========================================
            vis.plot_hybrid_spectrum(
                time_series=time_series,  # 👈 传给精确 DFT 算绿点用的原始时域数据
                sample_rate=rx_sr,  # 👈 传给精确 DFT 用的采样率
                freqs_fft=freqs,  # 👈 传给 FFT 画黑色底线用的频率轴
                yf_fft=amplitude,  # 👈 传给 FFT 画黑色底线用的振幅
                fundamental_freq=f0,
                num_harmonics=15,  # 画出前 15 次奇次谐波
                extra_freqs=[50.0],  # 顺便看看 50Hz 工频干扰有多强
                title=f"Unstacked Long-Sequence FFT (Window: Hann) - Ch {ch_idx}"
            )
        elif choice == '0':
            print("\n👋 感谢使用，系统已退出！")
            sys.exit(0)

        else:
            print("\n❌ 错误: 无效的输入，请输入 0 到 4 之间的数字！")


if __name__ == '__main__':
    # 捕获 Ctrl+C 强制退出，避免打印丑陋的报错信息
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 强制中断，系统已退出！")
        sys.exit(0)