import pandas as pd
import numpy as np
import argparse
import sys

def calculate_max_acceleration(file_path, dt=None):
    """
    读取真机离线轨迹 Excel 文件，并计算 q3 关节列的最大加速度。
    
    参数:
    file_path: Excel 文件路径
    dt: 采样控制周期（秒）。如果 Excel 中包含 'time' 或 't' 列，则会尝试自动计算均值 dt。
        如果指定了 dt 参数，则强制使用该控制周期。
    """
    print(f"正在读取文件: {file_path}")
    try:
        # 需要安装 pandas 和 openpyxl: pip install pandas openpyxl
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取 Excel 文件失败，请检查文件路径或确保已安装 openpyxl: {e}")
        sys.exit(1)
        
    print(f"✅ 成功读取数据，数据帧维度 (Time steps x Variables): {df.shape}")
    print(f"📊 包含的列名: {df.columns.tolist()}")
    
    # 尝试自动对齐观测的时间空间 (Time space)
    time_col = None
    for col in ['time', 't', 'Time', 'T']:
        if col in df.columns:
            time_col = col
            break
            
    if time_col and dt is None:
        # 使用时间列计算控制周期 dt (取差分均值)
        time_data = df[time_col].values
        dt = np.mean(np.diff(time_data))
        print(f"⏱️ 检测到时间序列列 '{time_col}'，自动估算的平均控制周期 dt = {dt:.6f} s (频率 ≈ {1/dt:.1f} Hz)")
    elif dt is None:
        # 默认假设控制频率为 50Hz -> dt = 0.02 (Unitree 等常见的高层策略频率)
        dt = 0.02
        print(f"⚠️ 未检测到明确的时间列，使用默认策略步长 dt = {dt} s。如需修改请通过 --dt 指定。")
    else:
        print(f"⚙️ 使用用户强制指定的控制周期 dt = {dt} s")
        
    # 检查是否存在 q3 关节列
    if 'q3' not in df.columns:
        print("❌ 错误: 观测数据中未找到 'q3' 列！请检查数据导出逻辑。")
        sys.exit(1)
        
    q3_data = df['q3'].values
    
    # 物理意义：使用中心差分计算关节速度 v = dq/dt
    # np.gradient 相比纯 np.diff 能保持数组长度一致，且在内部边界处使用前向/后向差分，中间使用中心差分
    velocity = np.gradient(q3_data, dt)
    
    # 物理意义：使用中心差分计算关节加速度 a = dv/dt
    acceleration = np.gradient(velocity, dt)
    
    # 分析极限物理状态
    max_acc = np.max(acceleration)         # 正向最大加速度
    min_acc = np.min(acceleration)         # 反向最大加速度 (减速/反转)
    max_abs_acc = np.max(np.abs(acceleration)) # 绝对加速度峰值
    
    print("\n" + "="*40)
    print("🎯 q3 关节加速度分析结果")
    print("="*40)
    print(f"正向极限加速度 (Max Positive Acc): {max_acc:.4f} rad/s^2")
    print(f"反向极限加速度 (Max Negative Acc): {min_acc:.4f} rad/s^2")
    print(f"绝对值峰值加速度 (Max Absolute Acc): {max_abs_acc:.4f} rad/s^2")
    print("="*40)
    print("提示：若此加速度远超真实电机的扭矩/带宽限制，建议在 RL 训练中加大 action_smoothness_penalty。")
    
    return max_abs_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算真机测试记录中的关节最大加速度 (Sim-to-Real Data Analysis)")
    # 默认路径指向用户提供的文件
    parser.add_argument("--file", type=str, default="test/真机关节轨迹.xlsx", help="Excel 数据文件路径")
    parser.add_argument("--dt", type=float, default=None, help="控制周期间隔大小 (秒)，未指定时尝试从表中读取时间差")
    args = parser.parse_args()
    
    calculate_max_acceleration(args.file, args.dt)