import subprocess
import time
import pandas as pd
import sys

# ================= 配置区域 =================
# 服务端 SSH 登录地址
SERVER_SSH_HOST = "10.11.0.6"

# 服务端的 TCP 握手 IP
SERVER_IP = "10.11.0.6"

# 设备列表 (假设两台机器都有这8张卡)
DEVICES = ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4", "mlx5_5", "mlx5_6", "mlx5_7"]

# 起始端口号 (BW测试通常持续时间稍长，用不同的段避免冲突)
BASE_PORT = 20000 
# ===========================================

results = []

print(f"Starting 8x8 IB Bandwidth Test (Socket Handshake via {SERVER_IP})...")
print("Note: Bandwidth tests take longer than latency tests.\n")

for client_idx, client_dev in enumerate(DEVICES):
    row_data = {}
    
    for server_idx, server_dev in enumerate(DEVICES):
        # 生成唯一端口
        current_port = BASE_PORT + (client_idx * len(DEVICES)) + server_idx
        
        print(f"Testing BW: Local[{client_dev}] -> Remote[{server_dev}] (Port: {current_port})")

        # 1. 远程启动 Server (ib_write_bw)
        server_cmd = f"ssh {SERVER_SSH_HOST} 'ib_write_bw -d {server_dev} -p {current_port}'"
        srv_proc = subprocess.Popen(server_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2) # 给 Server 多一点启动时间

        # 2. 本地启动 Client
        # -F: 忽略频率警告
        # --perform_warm_up: 预热
        # 默认包大小通常是 65536，足以打满带宽
        client_cmd = f"ib_write_bw -d {client_dev} -p {current_port} {SERVER_IP} --perform_warm_up -F"
        
        bw_val = -1.0
        
        try:
            # 运行客户端，BW测试通常需要几秒钟
            output = subprocess.check_output(client_cmd, shell=True, timeout=20).decode("utf-8")
            
            # 3. 解析结果
            # 典型输出格式:
            # #bytes #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
            # 65536  1000             192.50             191.20               ...
            
            lines = output.split('\n')
            for line in lines:
                parts = line.strip().split()
                # 寻找数据行：通常以 65536 开头 (默认msg size) 或者是数字开头且列数足够
                if len(parts) >= 4 and parts[0].isdigit():
                    try:
                        # ib_write_bw 的列顺序通常是:
                        # 0: #bytes
                        # 1: #iterations
                        # 2: BW peak
                        # 3: BW average <--- 我们要这个
                        # 4: MsgRate
                        
                        # 简单的校验：确保第一列是包大小 (通常 > 1000)
                        if int(parts[0]) > 1000:
                            bw_val = float(parts[3])
                            break
                    except (ValueError, IndexError):
                        continue
            
            if bw_val > 0:
                print(f"  -> Success. Avg BW: {bw_val} Mb/sec")
            else:
                print(f"  -> Failed to parse output.")

        except subprocess.CalledProcessError as e:
            print(f"  -> Client Error: {e}")
        except subprocess.TimeoutExpired:
            print(f"  -> Timeout.")
            srv_proc.kill()
        
        row_data[server_dev] = bw_val
        
        # 清理 Server 进程
        try:
            srv_proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            srv_proc.kill()
            
        time.sleep(0.5)

    results.append({"Local_Dev": client_dev, **row_data})

# ================= 结果输出 =================
print("\n" + "="*40)
print("Bandwidth Matrix (Mb/sec)")
print("Row: Local Client Device | Column: Remote Server Device")
print("="*40)

df = pd.DataFrame(results)
df.set_index("Local_Dev", inplace=True)

print(df)

csv_filename = "ib_bandwidth_matrix_8x8.csv"
df.to_csv(csv_filename)
print(f"\nResults saved to {csv_filename}")
