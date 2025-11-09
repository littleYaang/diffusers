#!/bin/bash

# 批量生成集群脚本
mkdir -p cluster_scripts

# 获取sh/eval/0729_cluster_eardiff/batch_scripts目录中所有脚本的数量
script_count=$(ls batch_scripts/run_*.sh 2>/dev/null | wc -l)

if [ $script_count -eq 0 ]; then
    echo "No scripts found in batch_scripts/"
    exit 1
fi

echo "Found $script_count scripts in batch_scripts/"

# 计算需要多少个集群 (每个集群4个脚本)
total_clusters=$(( (script_count + 3) / 4 ))
echo "Will generate $total_clusters cluster scripts"

# 生成集群脚本
for cluster_id in $(seq 0 $((total_clusters-1))); do
    script_name="cluster_${cluster_id}.sh"
    script_path="cluster_scripts/${script_name}"
    
    cat > "$script_path" << 'EOF'
#!/bin/bash

# 创建日志目录
mkdir -p /mnt/hero_blob/text_erase_lei/logs/eval_1109

idx=$(($1*4))
nohup bash sh/eval/0729_cluster_eardiff/batch_scripts/run_$(printf "%03d" $idx)_*.sh > /mnt/hero_blob/text_erase_lei/logs/eval_1109/${idx}.log 2>&1 &

idx=$(($1*4+1))
nohup bash sh/eval/0729_cluster_eardiff/batch_scripts/run_$(printf "%03d" $idx)_*.sh > /mnt/hero_blob/text_erase_lei/logs/eval_1109/${idx}.log 2>&1 &

idx=$(($1*4+2))
nohup bash sh/eval/0729_cluster_eardiff/batch_scripts/run_$(printf "%03d" $idx)_*.sh > /mnt/hero_blob/text_erase_lei/logs/eval_1109/${idx}.log 2>&1 &

idx=$(($1*4+3))
nohup bash sh/eval/0729_cluster_eardiff/batch_scripts/run_$(printf "%03d" $idx)_*.sh > /mnt/hero_blob/text_erase_lei/logs/eval_1109/${idx}.log 2>&1 &

wait
echo "Cluster $1 completed"
EOF
    
    chmod +x "$script_path"
    echo "Generated: $script_path"
done

echo ""
echo "Generated $total_clusters cluster scripts"
echo "Usage: ./cluster_scripts/cluster_0.sh 0"

#!/bin/bash

# 创建日志目录
mkdir -p /mnt/hero_blob/text_erase_lei/logs/eval_1109

idx=$(($1*4))
nohup bash generated_scripts/eval_${idx}_laion.sh > /mnt/hero_blob/text_erase_lei/logs/eval_1109/${idx}.log 2>&1 &

idx=$(($1*4+1))
nohup bash generated_scripts/eval_001_prim.sh > /mnt/hero_blob/text_erase_lei/logs/eval_1109/${idx}.log 2>&1 &

idx=$(($1*4+2))
nohup bash generated_scripts/eval_001_scut.sh > /mnt/hero_blob/text_erase_lei/logs/eval_1109/${idx}.log 2>&1 &

idx=$(($1*4+3))
nohup bash generated_scripts/eval_001_syn.sh > /mnt/hero_blob/text_erase_lei/logs/eval_1109/${idx}.log 2>&1 &
