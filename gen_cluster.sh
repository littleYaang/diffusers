#!/bin/bash
# generate_cluster_scripts_v2.sh

# 创建输出目录
mkdir -p cluster_scripts

echo "Generating 5 cluster scripts..."

# 生成5个脚本，对应模型 001-005
for model_id in {1..5}; do
    model_num=$(printf "%03d" $model_id)
    script_name="cluster_${model_num}.sh"
    script_path="cluster_scripts/${script_name}"
    
    cat > "$script_path" << 'SCRIPT_EOF'
#!/bin/bash

# 从脚本名称提取模型ID
SCRIPT_NAME=$(basename "$0")
MODEL_ID=${SCRIPT_NAME#cluster_}
MODEL_ID=${MODEL_ID%.sh}

# 创建日志目录
LOG_DIR="/mnt/hero_blob/text_erase_lei/logs/eval_1109"
mkdir -p ${LOG_DIR}

echo "========================================"
echo "Starting cluster for model ${MODEL_ID}"
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
    script_path="generated_scripts/eval_${MODEL_ID}_${dataset}.sh"
    log_file="${LOG_DIR}/${MODEL_ID}_${dataset}.log"
    
    if [ -f "$script_path" ]; then
        echo "Running eval_${MODEL_ID}_${dataset}.sh on GPU ${gpu_id}..."
        nohup bash "$script_path" > "$log_file" 2>&1 &
        echo "  PID: $! | Log: $log_file"
    else
        echo "Warning: $script_path not found, skipping..."
    fi
done

echo ""
echo "All 4 jobs for model ${MODEL_ID} submitted"
echo "Waiting for completion..."

# 等待所有后台任务完成
wait

echo ""
echo "========================================"
echo "Cluster for model ${MODEL_ID} completed"
echo "Completed at: $(date)"
echo "========================================"

# 显示日志文件位置
echo ""
echo "Log files:"
for dataset in laion prim scut syn; do
    log_file="${LOG_DIR}/${MODEL_ID}_${dataset}.log"
    if [ -f "$log_file" ]; then
        echo "  ${dataset}: $log_file"
    fi
done
SCRIPT_EOF
    
    # 替换脚本中的 MODEL_ID 占位符
    sed -i "s/\${MODEL_ID}/${model_num}/g" "$script_path"
    
    chmod +x "$script_path"
    echo "✓ Generated: $script_path"
done

echo ""
echo "✓ Generated 5 cluster scripts in cluster_scripts/"
echo ""
echo "Usage Examples:"
echo "  # Run model 001:"
echo "  ./cluster_scripts/cluster_001.sh"
echo ""
echo "  # Run all models sequentially:"
echo "  for i in {001..005}; do"
echo "    echo \"Starting model \$i...\""
echo "    ./cluster_scripts/cluster_\${i}.sh"
echo "    echo \"Model \$i completed\""
echo "    echo \"---\""
echo "  done"
echo ""
echo "  # Check logs:"
echo "  tail -f /mnt/hero_blob/text_erase_lei/logs/eval_1109/001_*.log"