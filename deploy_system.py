import os
import glob
import time
import json
import numpy as np
import torch
import joblib
import psutil
from pytorch_tabnet.tab_model import TabNetClassifier


# ==========================================

# ==========================================
def get_cpu_temperature():
    """获取当前 CPU 温度，兼容树莓派 Linux 底层文件系统与 psutil 接口"""
    try:
        # 首选方案：直接读取树莓派底层系统文件 (最稳定)
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp_c = float(f.read()) / 1000.0
            return temp_c
    except Exception:
        pass

    try:
        # 备用方案：使用 psutil 获取
        temps = psutil.sensors_temperatures()
        if 'cpu_thermal' in temps:
            return temps['cpu_thermal'][0].current
        elif 'coretemp' in temps:
            return temps['coretemp'][0].current
    except Exception:
        pass

    return 0.0  # 无法获取时返回 0.0


# ==========================================
# 0. 信号处理核心算法
# ==========================================
try:
    from main import load_signal_file, extract_features_from_signal

    print("[*] 成功挂载前端信号处理模块 (main.py)")
except ImportError:
    print("找不到 main.py！请确保 deploy_system.py 和 main.py 在同一个文件夹下。")
    exit()

# ==========================================
# 1. 绝对路径配置
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
optimized_model_dir = os.path.join(CURRENT_DIR, "optimized_comparison")
model_dir = os.getenv(
    "TABNET_MODEL_DIR",
    optimized_model_dir if os.path.exists(optimized_model_dir) else os.path.join(CURRENT_DIR, "model"),
)
raw_signal_dir = os.path.join(CURRENT_DIR, "signal")

print("=" * 60)
print("     树莓派全链路智能监测系统启动 ")
print("=" * 60)

# ==========================================
# 2. 开机一次性加载模型常驻内存
# ==========================================
print("\n[系统初始化] 正在将模型加载进内存中...")
if not os.path.exists(model_dir):
    print(f"\n找不到模型文件夹: {model_dir}")
    exit()

try:
    scaler_name = "feature_scaler.pkl" if os.path.exists(os.path.join(model_dir, "feature_scaler.pkl")) else "raw_feature_scaler.pkl"
    scaler = joblib.load(os.path.join(model_dir, scaler_name))
    feature_mask_path = os.path.join(model_dir, "feature_mask.npy")
    feature_mask = np.load(feature_mask_path) if os.path.exists(feature_mask_path) else None
    config_path = os.path.join(model_dir, "experiment_config.json")
    experiment_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            experiment_config = json.load(f)
    xgb_name = "xgboost_tabnet.pkl" if os.path.exists(os.path.join(model_dir, "xgboost_tabnet.pkl")) else "xgboost_hybrid.pkl"
    xgb_input = experiment_config.get("xgboost", {}).get("input_representation", "hybrid")
    decision_threshold = float(experiment_config.get("xgboost", {}).get("threshold", 0.5))
    clf_tabnet = TabNetClassifier(device_name="cpu")
    clf_tabnet.load_model(os.path.join(model_dir, "tabnet_encoder.zip"))
    clf_tabnet.network.eval()
    xgb_model = joblib.load(os.path.join(model_dir, xgb_name))
except Exception as e:
    print(f"\n 模型加载失败！\n错误详情: {e}")
    exit()

print(f"等待信号接入！")

if not os.path.exists(raw_signal_dir):
    os.makedirs(raw_signal_dir)
    print(f"\n已创建了 '{raw_signal_dir}' 文件夹。")
    print("        请将仪器采集到的【原始波形文件】放入其中，然后重新运行脚本！")
    exit()


# ==========================================
# 3. 核心流水线：原始信号 -> 内存提取特征 -> 直接推理
# ==========================================
def process_and_predict(raw_file_path):
    """端到端全链路处理 (拒绝硬盘读写中间文件)"""
    time_log = {}

    t_start = time.perf_counter()
    raw_signal = load_signal_file(raw_file_path)
    if raw_signal is None or len(raw_signal) == 0:
        return None, None

    try:
        features_17d = extract_features_from_signal(raw_signal, fs=100, scale_factor=1000)
    except Exception as e:
        print(f"提取特征失败: {e}")
        return None, None

    time_log['vmd_extraction'] = (time.perf_counter() - t_start) * 1000

    t_infer = time.perf_counter()
    feature_values = list(features_17d.values())
    X_input = np.array(feature_values).reshape(1, -1)
    if feature_mask is not None:
        X_input = X_input[:, feature_mask]
    X_scaled = scaler.transform(X_input)

    features = []

    def hook(module, input, output):
        features.append(input[0].cpu().detach().numpy())

    target_layer = clf_tabnet.network.final_mapping if hasattr(clf_tabnet.network, 'final_mapping') else \
    [m for m in clf_tabnet.network.modules() if isinstance(m, torch.nn.Linear)][-1]
    handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        _ = clf_tabnet.network(torch.tensor(X_scaled, dtype=torch.float32))
    handle.remove()
    tab_feat = np.concatenate(features, axis=0)

    X_model = tab_feat if xgb_input == "deep" else np.hstack([X_scaled, tab_feat])
    prediction_score = xgb_model.predict_proba(X_model)[0, 1]
    prediction = int(prediction_score >= decision_threshold)
    time_log['model_inference'] = (time.perf_counter() - t_infer) * 1000

    return prediction, time_log


# ==========================================
# 4. 开始扫描并处理
# ==========================================
signal_files = glob.glob(os.path.join(raw_signal_dir, "*.txt")) + glob.glob(os.path.join(raw_signal_dir, "*.csv"))

if len(signal_files) == 0:
    print(f"\n '{raw_signal_dir}' 文件夹为空，没有找到待处理的波形文件。")
    exit()

print(f"\n 发现 {len(signal_files)} 个原始波形信号，开始端到端检测...\n")

total_vmd_time = []
total_infer_time = []
total_cpu_usage = []
total_mem_usage = []
total_cpu_temp = []


current_process = psutil.Process(os.getpid())
peak_memory_mb = 0.0  # 初始化峰值内存记录仪

results_summary = {"泥石流": 0, "噪声": 0}

# 初始化 psutil CPU 监控基准
psutil.cpu_percent(interval=None)

for i, f_path in enumerate(signal_files):
    file_name = os.path.basename(f_path)

    # 1. 执行全链路预测
    pred, time_log = process_and_predict(f_path)

    # 2. 获取这段时间内的全局硬件指标
    current_cpu_usage = psutil.cpu_percent(interval=None)
    current_mem_usage = psutil.virtual_memory().percent
    current_cpu_temp = get_cpu_temperature()

    # 3. 获取当前进程的精确物理内存占用 (RSS)
    process_mem_mb = current_process.memory_info().rss / (1024 * 1024)
    if process_mem_mb > peak_memory_mb:
        peak_memory_mb = process_mem_mb  # 不断刷新历史最高记录

    if pred is None:
        continue

    if pred == 1:
        label_str = " 【泥石流预警】"
        results_summary["泥石流"] += 1
    else:
        label_str = "【安全噪声】"
        results_summary["噪声"] += 1

    # 记录硬件数据用于最后求平均
    total_cpu_usage.append(current_cpu_usage)
    total_mem_usage.append(current_mem_usage)
    if current_cpu_temp > 0:
        total_cpu_temp.append(current_cpu_temp)

    # 实时打印详情面板
    print(f"[{i + 1}/{len(signal_files)}] {file_name} -> {label_str}")
    print(f"    耗时: VMD提取 {time_log['vmd_extraction']:.1f}ms | AI推理 {time_log['model_inference']:.1f}ms")
    print(
        f"    硬件: CPU {current_cpu_usage:.1f}% | 算法独占内存 {process_mem_mb:.1f} MB | 温度 {current_cpu_temp:.1f}℃\n")

    total_vmd_time.append(time_log['vmd_extraction'])
    total_infer_time.append(time_log['model_inference'])

# ==========================================
# 5. 全链路性能报告
# ==========================================
if len(total_vmd_time) > 0:
    print("=" * 60)
    print(" 树莓派 4B 端到端全链路与硬件综合性能总结 ")
    print("=" * 60)
    print(f" : 识别出 {results_summary['泥石流']} 个泥石流, {results_summary['噪声']} 个噪声")
    print("-" * 60)
    print(f" 前端特征提取平均耗时 (VMD)  : {np.mean(total_vmd_time):.1f} ms")
    print(f" 后端异构模型平均耗时 (AI)   : {np.mean(total_infer_time):.1f} ms")
    print(f" 单次信号处理系统总延迟      : {np.mean(total_vmd_time) + np.mean(total_infer_time):.1f} ms")
    print("-" * 60)
    print(f" 均值 CPU 占用率 (防卡顿)    : {np.mean(total_cpu_usage):.1f} %")
    print(f" 均值 RAM 占用率 (系统总)    : {np.mean(total_mem_usage):.1f} %")

    # 输出系统在运行期间遇到的最大峰值内存消耗
    print(f"| 算法峰值物理内存开销(Max RSS): {peak_memory_mb:.1f} MB")

    avg_temp = np.mean(total_cpu_temp) if total_cpu_temp else 0.0
    print(f"| 均值 CPU 温度值 (防降频)    : {avg_temp:.1f} ℃")
    print("=" * 60)

    total_latency = np.mean(total_vmd_time) + np.mean(total_infer_time)
    print("\n[建议]:")

    if total_latency <= 1000:
        print("  实时性达标：端到端延迟小于1秒，满足地质灾害瞬时预警需求。")
    else:
        print("  实时性警告：总耗时较长，建议优化前端 VMD 分解层数或下采样率。")

    if peak_memory_mb < 500:
        print(f" 内存表现极佳：峰值独立内存仅占用 {peak_memory_mb:.1f} MB，极其契合边缘设备的部署环境！")

    if avg_temp > 75:
        print("  散热告警：核心温度过高（>75℃），存在过热降频风险，请加装主动散热风扇！")
