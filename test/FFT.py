import numpy as np
from scipy import signal

def calculate_phase_lag(target, actual, fs, excitation_freq):
    """
    计算实际曲线相对于目标曲线的相位滞后
    :param target: 目标曲线数组 (numpy array)
    :param actual: 实际跟踪曲线数组 (numpy array)
    :param fs: 采样频率 (Hz)，例如 1000 (1ms控制周期)
    :param excitation_freq: 发送正弦指令的频率 (Hz)
    :return: 互相关法计算的滞后角, FFT法计算的滞后角
    """
    # 确保去除直流偏置（均值），否则会严重影响互相关和FFT的结果
    target = target - np.mean(target)
    actual = actual - np.mean(actual)
    
    # ---------------------------------------------------------
    # 方法 A: 互相关法 (Cross-Correlation)
    # ---------------------------------------------------------
    # 计算互相关
    correlation = signal.correlate(actual, target, mode='full')
    # 互相关的零延迟点在数组的正中间
    lags = signal.correlation_lags(len(actual), len(target), mode='full')
    
    # 找到最大相关性对应的平移 index
    lag_idx = lags[np.argmax(correlation)]
    
    # 计算时间延迟和相位滞后
    time_delay_corr = lag_idx / fs
    phase_lag_corr_deg = (time_delay_corr * excitation_freq) * 360.0
    
    # ---------------------------------------------------------
    # 方法 B: 频域 FFT 法
    # ---------------------------------------------------------
    # 对信号做 FFT
    target_fft = np.fft.rfft(target)
    actual_fft = np.fft.rfft(actual)
    freqs = np.fft.rfftfreq(len(target), d=1/fs)
    
    # 找到与激励频率最接近的频率 bin 索引
    idx = np.argmin(np.abs(freqs - excitation_freq))
    
    # 提取该频率下的复数向量
    H_complex = actual_fft[idx] / target_fft[idx]
    
    # 计算相角 (弧度)，然后转角度
    phase_rad = np.angle(H_complex)
    # 相位滞后通常定义为正数，所以取负
    phase_lag_fft_deg = -np.degrees(phase_rad)
    
    # 处理相位卷绕 (Wrap)，确保在 0~180 度之间
    phase_lag_fft_deg = phase_lag_fft_deg % 360
    if phase_lag_fft_deg > 180:
        phase_lag_fft_deg -= 360
        
    return phase_lag_corr_deg, phase_lag_fft_deg

# ================= 模拟测试用例 =================
if __name__ == "__main__":
    fs = 1000.0          # 控制频率 1000Hz
    f_test = 2.0         # 激励频率 2Hz
    t = np.arange(0, 5, 1/fs) # 5秒的数据
    
    # 生成测试数据：假设存在 110ms 的纯延迟，且实际幅值衰减为 0.8，并加入随机高斯噪声
    delay_ms = 110
    true_phase_lag = (delay_ms / 1000.0) * f_test * 360 # 应该等于 79.2 度
    
    target_sig = np.sin(2 * np.pi * f_test * t)
    # 实际信号：相位滞后，幅值衰减，外加噪声
    actual_sig = 0.8 * np.sin(2 * np.pi * f_test * t - np.radians(true_phase_lag)) 
    actual_sig += np.random.normal(0, 0.1, len(t))
    
    lag_corr, lag_fft = calculate_phase_lag(target_sig, actual_sig, fs, f_test)
    
    print(f"理论设定滞后角: {true_phase_lag:.2f} 度 (等效 {delay_ms} ms)")
    print(f"互相关法计算结果: {lag_corr:.2f} 度")
    print(f"FFT法计算结果: {lag_fft:.2f} 度")