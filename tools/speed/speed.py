import subprocess
import psutil
import csv
import time
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
)


def monitor_resources(pid, interval=0.2, output_file="resource_usage.csv"):
    """监控指定进程的CPU、内存、GPU使用情况"""
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(device_count)] if device_count > 0 else []

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_sec",  # 修改后字段名
            "cpu_percent",
            "memory_rss_mb",
            "gpu_mem_used_mb",
            "gpu_util_percent",
        ])

        process = psutil.Process(pid)
        elapsed_time = 0  # 新增时间计数器
        while process.is_running():  # 更可靠的退出条件
            try:
                # 获取CPU和内存
                cpu_percent = process.cpu_percent(interval=None)
                memory_info = process.memory_info()
                memory_rss_mb = memory_info.rss / 1024 ** 2

                # 获取GPU数据（取第一个GPU）
                gpu_mem_used_mb = 0
                gpu_util_percent = 0
                if device_count > 0:
                    mem_info = nvmlDeviceGetMemoryInfo(gpu_handles[0])
                    util = nvmlDeviceGetUtilizationRates(gpu_handles[0])
                    gpu_mem_used_mb = mem_info.used / 1024 ** 2
                    gpu_util_percent = util.gpu

                # 写入数据
                writer.writerow([
                    elapsed_time,  # 使用递增时间
                    cpu_percent,
                    memory_rss_mb,
                    gpu_mem_used_mb,
                    gpu_util_percent,
                ])
                f.flush()

                elapsed_time += interval  # 时间递增
                time.sleep(interval)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    nvmlShutdown()


if __name__ == "__main__":
    # 启动命令
    cmd = [
        "python", "/home/pointseg/Yolov8/tools/speed/modelPredict.py"
    ]
    process = subprocess.Popen(cmd)

    # 启动监控
    monitor_resources(process.pid)

    # 等待命令执行完成
    process.wait()
