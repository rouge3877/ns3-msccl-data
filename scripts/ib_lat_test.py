import subprocess
import time
import pandas as pd
import sys

# ================= 配置区域 =================
# 服务端 SSH 登录地址 (用于远程启动命令)
SERVER_SSH_HOST = "10.11.0.6"

# 服务端的 TCP 握手 IP (所有连接都指向这个 IP)
SERVER_IP = "10.11.0.6"

# 设备列表 (假设两台机器都有这8张卡)
DEVICES = ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4", "mlx5_5", "mlx5_6", "mlx5_7"]

# 起始端口号 (避免使用常见端口，从 10000 开始)
BASE_PORT = 19000
# ===========================================

results = []

print(f"Starting 8x8 IB Latency Test (Socket Handshake via {SERVER_IP})...")
print(f"Total tests to run: {len(DEVICES) * len(DEVICES)}\n")

# 两个循环：遍历本地8张卡 vs 远端8张卡
for client_idx, client_dev in enumerate(DEVICES):
    row_data = {}
    
    for server_idx, server_dev in enumerate(DEVICES):
        # 为每一对连接生成一个唯一的端口号
        # 公式：Base + (Client索引 * 8) + Server索引
        # 这样可以确保 0-0, 0-1 ... 7-7 所有的端口都不一样
        current_port = BASE_PORT + (client_idx * len(DEVICES)) + server_idx
        
        print(f"Testing: Local[{client_dev}] -> Remote[{server_dev}] (Port: {current_port})")

        # 1. 远程启动 Server
        # 注意：服务端监听在指定端口，并绑定到特定的 IB 设备 (-d)
        server_cmd = f"ssh {SERVER_SSH_HOST} 'ib_write_lat -d {server_dev} -p {current_port}'"
        
        # 异步启动 Server，不等待它完成，因为通过 SSH 启动会阻塞直到测试结束
        srv_proc = subprocess.Popen(server_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待 1 秒确保 Server 端的 Socket 已经 Listen 起来了
        time.sleep(1.5)

        # 2. 本地启动 Client
        # 连接到 SERVER_IP，使用对应的端口，并绑定本地 IB 设备 (-d)
        # --perform_warm_up: 预热，避免第一次延迟过高
        # -F: 忽略 CPU 频率警告格式
        client_cmd = f"ib_write_lat -d {client_dev} -p {current_port} {SERVER_IP} --perform_warm_up -F"
        
        latency_val = -1.0 # 默认失败值
        
        try:
            # 运行客户端并捕获输出
            output = subprocess.check_output(client_cmd, shell=True, timeout=15).decode("utf-8")
            
            # 3. 解析结果
            # 寻找包含数据的那一行，通常特征是 "2    1000 ..."
            lines = output.split('\n')
            for line in lines:
                parts = line.strip().split()
                # 检查是否是数据行：第一列是bytes(2)，且列数足够
                if len(parts) >= 6 and parts[0] == '2':
                    # ib_write_lat 输出列: bytes, iterations, t_min, t_max, t_typical, t_avg ...
                    # t_avg 通常是第 6 列 (索引 5)
                    try:
                        latency_val = float(parts[5])
                    except ValueError:
                        pass
                    break
            
            if latency_val > 0:
                print(f"  -> Success. Avg Latency: {latency_val} us")
            else:
                print(f"  -> Failed to parse output.")

        except subprocess.CalledProcessError as e:
            print(f"  -> Client Error: {e}")
        except subprocess.TimeoutExpired:
            print(f"  -> Timeout. Cleaning up.")
            # 如果客户端超时，杀死 SSH 进程
            srv_proc.kill()
        
        # 记录数据
        row_data[server_dev] = latency_val
        
        # 确保 Server 进程结束 (通常 Client 跑完 Server 会自动退出，但为了保险)
        try:
            srv_proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            srv_proc.kill()
            
        # 冷却时间，让 TCP 端口释放
        time.sleep(0.2)

    results.append({"Local_Dev": client_dev, **row_data})

# ================= 结果输出 =================
print("\n" + "="*40)
print("Result Matrix (Latency in microseconds)")
print("Row: Local Client Device | Column: Remote Server Device")
print("="*40)

df = pd.DataFrame(results)
df.set_index("Local_Dev", inplace=True)

# 打印到屏幕
print(df)

# 保存到 CSV，方便你在仿真软件中导入
csv_filename = "ib_latency_matrix_8x8.csv"
df.to_csv(csv_filename)
print(f"\nResults saved to {csv_filename}")
