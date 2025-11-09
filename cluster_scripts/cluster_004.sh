#!/bin/bash

# 从脚本名称提取模型ID
SCRIPT_NAME=$(basename "$0")
MODEL_ID=${SCRIPT_NAME#cluster_}
MODEL_ID=${MODEL_ID%.sh}

# 创建日志目录
LOG_DIR="/mnt/hero_blob/text_erase_lei/logs/eval_1109"
mkdir -p ${LOG_DIR}

echo "========================================"
echo "Starting cluster for model 004"
echo "Started at: $(date)"
echo "========================================"

# 定义数据集和对应的GPU
declare -A datasets=(
    ["laion"]=2
    ["prim"]=3
    ["scut"]=0
    ["syn"]=1
)

# 启动所有任务
for dataset in "${!datasets[@]}"; do
    gpu_id=${datasets[$dataset]}
    script_path="generated_scripts/eval_004_${dataset}.sh"
    log_file="${LOG_DIR}/004_${dataset}.log"
    
    if [ -f "$script_path" ]; then
        echo "Running eval_004_${dataset}.sh on GPU ${gpu_id}..."
        nohup bash "$script_path" > "$log_file" 2>&1 &
        echo "  PID: $! | Log: $log_file"
    else
        echo "Warning: $script_path not found, skipping..."
    fi
done

echo ""
echo "All 4 jobs for model 004 submitted"
echo "Waiting for completion..."

# 等待所有后台任务完成
wait

echo ""
echo "========================================"
echo "Cluster for model 004 completed"
echo "Completed at: $(date)"
echo "========================================"

# 显示日志文件位置
echo ""
echo "Log files:"
for dataset in laion prim scut syn; do
    log_file="${LOG_DIR}/004_${dataset}.log"
    if [ -f "$log_file" ]; then
        echo "  ${dataset}: $log_file"
    fi
done
