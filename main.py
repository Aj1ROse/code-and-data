#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
泥石流信号特征提取工具
使用方法: python main.py <信号文件路径> [输出文件路径] [缩放因子]
"""

import numpy as np
import sys
import time
import os
from pathlib import Path
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis, pearsonr
import pywt
from VMD import VMD

# 固定VMD参数
FIXED_ALPHA = 944
FIXED_K = 6
FIXED_THRESHOLD = 0.4

def load_signal_file(file_path):
    """加载信号文件"""
    try:
        data = np.loadtxt(file_path, delimiter=',')
        if data.ndim == 2:
            return data[:, 1]
        else:
            return data
    except:
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 2:
                return data[:, 1]
            else:
                return data
        except:
            return None

def extract_features_from_signal(signal_data, fs=100, scale_factor=0.305):
    """从信号中提取特征值"""
    
    # 统一单位转换（根据信号类型使用不同的缩放因子）
    signal_data = signal_data * scale_factor
    
    # 信号中心化
    signal_data = signal_data - np.mean(signal_data)
    
    # 带通滤波 (0.5-20Hz)
    order = 5
    Wn = np.array([0.5, 20]) / (fs / 2)
    b, a = butter(order, Wn, btype='bandpass')
    filtered_signal = filtfilt(b, a, signal_data)

    # 小波降噪
    coeffs = pywt.wavedec(filtered_signal, 'sym8', level=3)
    n = len(filtered_signal)
    denoised_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        sigma_level = np.median(np.abs(coeffs[i])) / 0.6745
        thresh_level = sigma_level * np.sqrt(2 * np.log(n))
        denoised_coeffs.append(pywt.threshold(coeffs[i], thresh_level, mode='soft'))

    denoised_signal = pywt.waverec(denoised_coeffs, 'sym8')
    if len(denoised_signal) != len(filtered_signal):
        denoised_signal = denoised_signal[:len(filtered_signal)]

    # VMD分解
    DATA = {
        'data': denoised_signal,
        'tau': 0,
        'dc': 0,
        'init': 1,
        'tol': 1e-7
    }

    u, u_hat, omega = VMD(DATA['data'], FIXED_ALPHA, DATA['tau'],
                          FIXED_K, DATA['dc'], DATA['init'], DATA['tol'])

    # 双指标综合评价
    m, n = u.shape
    corr_coeffs = np.zeros(m)
    energy_contributions = np.zeros(m)
    
    for i in range(m):
        # 时域相关性
        corr_coeffs[i] = abs(pearsonr(u[i, :], DATA['data'])[0])

        # 频域能量贡献 (5-15Hz)
        freqs, psd = welch(u[i, :], fs, nperseg=min(256, len(u[i, :]) // 4))
        target_band_idx = (freqs >= 5) & (freqs <= 15)
        target_band_energy = np.sum(psd[target_band_idx])
        total_energy = np.sum(psd)
        energy_contributions[i] = target_band_energy / (total_energy + 1e-10)

    # 综合评分
    energy_coeffs_norm = energy_contributions / (np.max(energy_contributions) + 1e-10)
    comprehensive_scores = 0.6 * corr_coeffs + 0.4 * energy_coeffs_norm

    # 信号重构
    reconstructed_signal = np.zeros(n)
    for i in range(m):
        if comprehensive_scores[i] >= FIXED_THRESHOLD:
            reconstructed_signal += u[i, :]

    # 提取17个特征值
    features = {}
    signal = reconstructed_signal

    # 时域特征
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['rms'] = np.sqrt(np.mean(signal ** 2))
    features['peak'] = np.max(np.abs(signal))
    features['crest'] = np.max(np.abs(signal)) / np.sqrt(np.mean(signal ** 2))
    features['skewness'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)
    features['impulse'] = np.max(np.abs(signal)) / np.mean(np.abs(signal))
    features['shape_factor'] = np.sqrt(np.mean(signal ** 2)) / np.mean(np.abs(signal))
    features['amplitude_mean'] = np.mean(np.abs(signal))
    features['amplitude_max'] = np.max(np.abs(signal))

    # 频域特征
    freq, psd = welch(signal, fs)
    features['dominant_freq'] = freq[np.argmax(psd)]
    features['mean_freq'] = np.sum(freq * psd) / np.sum(psd)
    features['median_freq'] = np.median(freq)
    psd_norm = psd / np.sum(psd)
    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + np.finfo(float).eps)) / np.log2(len(psd))
    features['spectral_energy_mean'] = np.mean(psd)
    
    band_5_15_idx = (freq >= 5) & (freq <= 15)
    band_5_15_energy = np.sum(psd[band_5_15_idx])
    total_energy = np.sum(psd)
    features['band_5_15_energy_ratio'] = band_5_15_energy / total_energy

    return features

def process_signal_file(input_file, output_file=None, scale_factor=1000, verbose=True):
    """处理单个信号文件"""
    
    # 检查输入文件
    if not os.path.exists(input_file):
        if verbose:
            print(f"错误: 找不到输入文件 {input_file}")
        return False
    
    # 生成输出文件名
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_features.csv"
    
    try:
        # 加载信号
        signal_data = load_signal_file(input_file)
        if signal_data is None or len(signal_data) == 0:
            if verbose:
                print(f"错误: 无法加载信号数据")
            return False
        
        # 检查信号长度是否足够
        if len(signal_data) < 100:  # 需要足够的采样点进行分析
            if verbose:
                print(f"错误: 信号长度太短 ({len(signal_data)} 采样点)")
            return False
        
        if verbose:
            print(f"信号长度: {len(signal_data)} 个采样点")
            print("正在提取特征值...")
        
        # 提取特征值
        features = extract_features_from_signal(signal_data, scale_factor=scale_factor)
        
        # 检查特征值是否有效
        if not features or any(not np.isfinite(v) for v in features.values()):
            if verbose:
                print(f"错误: 提取的特征值无效")
            return False

        # 保存特征值到CSV文件
        feature_names = list(features.keys())
        feature_values = list(features.values())
        feature_header = ','.join(feature_names)
        
        np.savetxt(output_file, [feature_values], delimiter=',', 
                  header=feature_header, comments='')
        
        if verbose:
            print(f"特征提取完成: {output_file}")
        return True
        
    except Exception as e:
        if verbose:
            print(f"处理错误: {str(e)}")
        return False



def print_usage():
    """打印使用说明"""
    print("=" * 50)
    print(" 信号特征提取工具")
    print("=" * 50)
    print("使用方法:")
    print("  python main.py <信号文件路径> [输出文件路径] [缩放因子]")
    print("")
    print("参数说明:")
    print("  <信号文件路径>  - 必需，输入的信号文件(.txt)")
    print("  [输出文件路径]  - 可选，输出的特征文件(.csv)")
    print("                   如不指定，自动生成为 原文件名_features.csv")
    print("  [缩放因子]      - 可选，信号缩放因子 (默认: 1000)")
    print("                   噪声信号推荐: 0.305")
    print("                   泥石流信号推荐: 1000")
    print("")
    print("示例:")
    print("  python main.py signal.txt")
    print("  python main.py signal.txt features.csv")
    print("  python main.py signal.txt features.csv 0.305")
    print("  python main.py /path/to/signal.txt /path/to/output.csv 1000")
    print("")
    print("固定参数:")
    print(f"  VMD参数: Alpha={FIXED_ALPHA}, K={FIXED_K}")
    print(f"  带通滤波: 0.5-20Hz")
    print(f"  小波降噪: sym8, 3层分解")
    print("=" * 50)

def main():
    """主程序"""
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print_usage()
        return
    
    if sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        return
    
    # 开始计时
    start_time = time.time()
    
    # 解析命令行参数
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    scale_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    print(f" 输入文件: {input_file}")
    if output_file:
        print(f" 输出文件: {output_file}")
    else:
        print(f" 输出文件: 自动生成")
    print(f" 缩放因子: {scale_factor}")
    
    # 处理信号文件
    success = process_signal_file(input_file, output_file, scale_factor=scale_factor, verbose=True)

    # 计算总耗时
    total_time = time.time() - start_time
    
    if success:
        print(f"处理成功！总耗时: {total_time:.3f}秒")
    else:
        print(f"处理失败！总耗时: {total_time:.3f}秒")
        sys.exit(1)

if __name__ == "__main__":
    main() 