#!/usr/bin/env python3
import os
import glob

def generate_cluster_scripts():
    """生成集群批处理脚本"""
    
    # 配置路径
    batch_scripts_dir = "sh/eval/0722_cluster/batch_scripts"
    cluster_scripts_dir = "sh/eval/0722_cluster/cluster_scripts"
    log_dir = "/mnt/hero_blob/text_erase_lei/logs/eval_0722"

    # 创建目录
    # os.makedirs(cluster_scripts_dir, exist_ok=True)
    # os.makedirs(log_dir, exist_ok=True)
    
    # 获取所有批处理脚本
    pattern = os.path.join(batch_scripts_dir, "run_*.sh")
    all_scripts = sorted(glob.glob(pattern))
    
    if not all_scripts:
        print(f"No scripts found in {batch_scripts_dir}")
        return
    
    print(f"Found {len(all_scripts)} scripts")
    
    # 每个集群处理4个脚本
    scripts_per_cluster = 4
    total_clusters = (len(all_scripts) + scripts_per_cluster - 1) // scripts_per_cluster
    
    print(f"Will create {total_clusters} cluster scripts")
    
    # 生成集群脚本
    for cluster_id in range(total_clusters):
        cluster_script_path = os.path.join(cluster_scripts_dir, f"cluster_{cluster_id}.sh")
        
        # 计算当前集群要处理的脚本索引
        start_idx = cluster_id * scripts_per_cluster
        end_idx = min(start_idx + scripts_per_cluster, len(all_scripts))
        
        cluster_scripts = all_scripts[start_idx:end_idx]
        
        script_content = f"""#!/bin/bash

# 集群脚本 {cluster_id} - 处理 {len(cluster_scripts)} 个任务
# 生成时间: $(date)

# 创建日志目录
mkdir -p {log_dir}

echo "Starting cluster {cluster_id} with {len(cluster_scripts)} tasks at $(date)"
echo "Processing scripts {start_idx} to {end_idx-1}"

"""
        
        # 为每个脚本添加启动命令
        for i, script_path in enumerate(cluster_scripts):
            script_name = os.path.basename(script_path)
            script_idx = start_idx + i
            
            script_content += f"""
# 任务 {i+1}/{len(cluster_scripts)} - Script: {script_name}
echo "Starting task {script_idx}: {script_name}"
nohup bash {script_path} > {log_dir}/{script_idx:03d}.log 2>&1 &
"""
        
        script_content += f"""
# 等待所有后台任务完成
wait

echo "Cluster {cluster_id} completed at $(date)"
echo "All {len(cluster_scripts)} tasks finished successfully"
"""
        
        # 写入脚本文件
        with open(cluster_script_path, 'w') as f:
            f.write(script_content)
        
        # 设置执行权限
        os.chmod(cluster_script_path, 0o755)
        print(f"Generated: {cluster_script_path}")
    
    # 生成主启动脚本
    master_script_path = os.path.join(cluster_scripts_dir, "start_all_clusters.sh")
    master_content = f"""#!/bin/bash

# 主启动脚本 - 启动所有集群
# 生成时间: $(date)

echo "========================================"
echo "Starting all clusters for TextPaint evaluation"
echo "Total clusters: {total_clusters}"
echo "Total scripts: {len(all_scripts)}"
echo "Scripts per cluster: {scripts_per_cluster}"
echo "========================================"

# 创建日志目录
mkdir -p {log_dir}

"""
    
    for cluster_id in range(total_clusters):
        master_content += f"""
echo "Starting cluster {cluster_id}..."
bash {cluster_scripts_dir}/cluster_{cluster_id}.sh &
sleep 2  # 短暂等待避免同时启动过多进程
"""
    
    master_content += f"""
echo "All clusters started. Waiting for completion..."

# 等待所有集群完成
wait

echo "========================================"
echo "All clusters completed at $(date)"
echo "Check logs in: {log_dir}"
echo "========================================"
"""
    
    with open(master_script_path, 'w') as f:
        f.write(master_content)
    
    os.chmod(master_script_path, 0o755)
    print(f"Generated master script: {master_script_path}")
    
    # 生成README
    readme_content = f"""# 集群批处理脚本使用说明

## 生成的文件

- **集群脚本**: `cluster_0.sh` to `cluster_{total_clusters-1}.sh` - 每个集群处理 {scripts_per_cluster} 个任务
- **主启动脚本**: `start_all_clusters.sh` - 启动所有集群
- **日志目录**: `{log_dir}` - 存储所有执行日志

## 脚本分布

- 总脚本数: {len(all_scripts)}
- 集群数: {total_clusters}
- 每集群处理: {scripts_per_cluster} 个脚本

## 使用方法

### 方法1: 启动单个集群
```bash
cd {cluster_scripts_dir}
./cluster_0.sh  # 启动集群0
./cluster_1.sh  # 启动集群1
# ... 依此类推
```

### 方法2: 启动所有集群
```bash
cd {cluster_scripts_dir}
./start_all_clusters.sh
```

### 方法3: 分批启动（推荐）
```bash
cd {cluster_scripts_dir}
# 启动前几个集群
./cluster_0.sh &
./cluster_1.sh &
./cluster_2.sh &

# 等待一段时间后启动更多
sleep 300
./cluster_3.sh &
./cluster_4.sh &
```

## 监控和日志

- 实时查看日志: `tail -f {log_dir}/000.log`
- 检查所有日志: `ls -la {log_dir}/`
- 查看集群状态: `ps aux | grep cluster`

## 注意事项

1. 确保有足够的GPU资源
2. 监控磁盘空间和内存使用
3. 建议分批启动集群，避免系统过载
4. 每个脚本都会生成独立的日志文件

## 预估时间

- 单个脚本: 30-90分钟
- 单个集群: 2-6小时
- 全部完成: 根据并行度而定
"""
    
    readme_path = os.path.join(cluster_scripts_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Generated README: {readme_path}")
    
    print(f"\n======================================")
    print(f"✓ Generated {total_clusters} cluster scripts")
    print(f"✓ Generated master startup script")
    print(f"✓ Generated documentation")
    print(f"======================================")
    print(f"To start all clusters: ")
    print(f"  cd {cluster_scripts_dir}")
    print(f"  ./start_all_clusters.sh")
    print(f"======================================")


if __name__ == "__main__":
    generate_cluster_scripts()