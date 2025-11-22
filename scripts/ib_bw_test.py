import subprocess
import time
import pandas as pd
import sys

# ================= 配置区域 =================
# 服务端 SSH 登录地址
SERVER_SSH_HOST = "10.11.0.6"

# 服务端的 TCP 握手 IP
SERVER_IP = "10.11.0.6"

# 本地节点的 IP (用于本地客户端测试本地服务端)
LOCAL_IP = "127.0.0.1"

# 本地节点的 IP (用于远程客户端访问本地服务端) - 需要填写本机实际的网络IP
LOCAL_IP_FOR_REMOTE = "10.11.0.9"  # 请修改为本机的实际IP地址，例如 "10.11.0.5"

# 设备列表 (假设两台机器都有这8张卡)
DEVICES = ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4", "mlx5_5", "mlx5_6", "mlx5_7"]

# 起始端口号 (BW测试通常持续时间稍长，用不同的段避免冲突)
BASE_PORT = 22000 
# ===========================================

results = []

print(f"Starting 16x16 IB Bandwidth Test (Complete matrix test)...")
print("Note: Bandwidth tests take longer than latency tests.\n")
print(f"Total tests to run: {len(DEVICES) * 2 * len(DEVICES) * 2}\n")

# 测试函数
def test_bandwidth(client_dev, server_dev, server_ip, client_is_remote, server_is_remote, port_offset):
    current_port = BASE_PORT + port_offset
    client_loc = "Remote" if client_is_remote else "Local"
    server_loc = "Remote" if server_is_remote else "Local"
    print(f"Testing BW: {client_loc}[{client_dev}] -> {server_loc}[{server_dev}] (Port: {current_port})")
    
    # 1. 启动服务端
    if server_is_remote:
        # 远程服务端
        server_cmd = f"ssh {SERVER_SSH_HOST} 'ib_write_bw -d {server_dev} -p {current_port}'"
        srv_proc = subprocess.Popen(server_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        # 本地服务端
        server_cmd = f"ib_write_bw -d {server_dev} -p {current_port}"
        srv_proc = subprocess.Popen(server_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(2)
    
    # 2. 启动客户端
    bw_val = -1.0
    
    if client_is_remote:
        # 远程客户端：通过SSH在远程机器上执行客户端命令
        client_cmd = f"ssh {SERVER_SSH_HOST} 'ib_write_bw -d {client_dev} -p {current_port} {server_ip} --perform_warm_up -F'"
        
        try:
            output = subprocess.check_output(client_cmd, shell=True, timeout=20).decode("utf-8")
            lines = output.split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0].isdigit():
                    try:
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
    else:
        # 本地客户端
        client_cmd = f"ib_write_bw -d {client_dev} -p {current_port} {server_ip} --perform_warm_up -F"
        
        try:
            output = subprocess.check_output(client_cmd, shell=True, timeout=20).decode("utf-8")
            lines = output.split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0].isdigit():
                    try:
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
    
    # 3. 清理服务端进程
    try:
        srv_proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        srv_proc.kill()
    
    time.sleep(0.5)
    return bw_val

# 测试本地8张卡作为客户端
for client_idx, client_dev in enumerate(DEVICES):
    row_data = {}
    
    # 先测试本地8张卡作为服务端
    for server_idx, server_dev in enumerate(DEVICES):
        port_offset = (client_idx * len(DEVICES) * 2) + server_idx
        bw_val = test_bandwidth(client_dev, server_dev, LOCAL_IP, 
                               client_is_remote=False, server_is_remote=False, 
                               port_offset=port_offset)
        row_data[f"Local_{server_dev}"] = bw_val
    
    # 再测试远程8张卡作为服务端
    for server_idx, server_dev in enumerate(DEVICES):
        port_offset = (client_idx * len(DEVICES) * 2) + len(DEVICES) + server_idx
        bw_val = test_bandwidth(client_dev, server_dev, SERVER_IP, 
                               client_is_remote=False, server_is_remote=True, 
                               port_offset=port_offset)
        row_data[f"Remote_{server_dev}"] = bw_val

    results.append({"Client_Dev": f"Local_{client_dev}", **row_data})

# 测试远程8张卡作为客户端
for client_idx, client_dev in enumerate(DEVICES):
    row_data = {}
    
    # 远程客户端 -> 本地8张卡作为服务端
    for server_idx, server_dev in enumerate(DEVICES):
        port_offset = ((len(DEVICES) + client_idx) * len(DEVICES) * 2) + server_idx
        bw_val = test_bandwidth(client_dev, server_dev, LOCAL_IP_FOR_REMOTE, 
                               client_is_remote=True, server_is_remote=False, 
                               port_offset=port_offset)
        row_data[f"Local_{server_dev}"] = bw_val
    
    # 远程客户端 -> 远程8张卡作为服务端
    for server_idx, server_dev in enumerate(DEVICES):
        port_offset = ((len(DEVICES) + client_idx) * len(DEVICES) * 2) + len(DEVICES) + server_idx
        bw_val = test_bandwidth(client_dev, server_dev, SERVER_IP, 
                               client_is_remote=True, server_is_remote=True, 
                               port_offset=port_offset)
        row_data[f"Remote_{server_dev}"] = bw_val

    results.append({"Client_Dev": f"Remote_{client_dev}", **row_data})

# ================= 结果输出 =================
print("\n" + "="*60)
print("Bandwidth Matrix (Mb/sec)")
print("Row: Local + Remote Client Devices | Column: Local + Remote Server Devices")
print("="*60)

df = pd.DataFrame(results)
df.set_index("Client_Dev", inplace=True)

print(df)

csv_filename = "ib_bandwidth_matrix_16x16.csv"
df.to_csv(csv_filename)
print(f"\nResults saved to {csv_filename}")
print(f"Matrix size: 16 (8 local + 8 remote clients) x 16 (8 local + 8 remote servers)")
print("Complete 16x16 matrix generated from single execution.")
