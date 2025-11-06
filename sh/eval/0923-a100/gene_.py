import os
import itertools
from pathlib import Path

def generate_eval_scripts():
    """生成不同参数组合的评估脚本"""
    
    # 定义参数组合
    guidance_scales = [5.0]
    model_paths = [
        "/mnt/hero_blob/preweight/Qwen/Qwen-Image",
        "/mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit", 
        "/mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509"
    ]
    
    # 固定参数
    base_config = {
        "input_dir": "/mnt/hero_blob/DATASET/SynthText/SCUT-EnsText_test",
        "output_base_dir": "/mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test",
        "prompt": "remove the text",
        "num_inference_steps": 50,
        "batch_size": 1
    }
    
    # 创建输出目录
    scripts_dir = Path("eval_scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # 生成所有组合的脚本
    script_count = 0
    all_scripts = []
    
    for guidance_scale, model_path in itertools.product(guidance_scales, model_paths):
        script_count += 1
        
        # 从模型路径提取模型名称
        model_name = Path(model_path).name
        if model_name == "Qwen-Image":
            model_short_name = "base"
        elif "Edit-2509" in model_name:
            model_short_name = "edit2509"
        elif "Edit" in model_name:
            model_short_name = "edit"
        else:
            model_short_name = model_name
        
        # 构建输出目录
        output_dir = f"{base_config['output_base_dir']}/{model_short_name}_gs{guidance_scale}"
        
        # 生成脚本内容
        script_content = f"""#!/bin/bash
# Evaluation script for {model_short_name} with guidance_scale={guidance_scale}
# Generated automatically

echo "Starting evaluation for {model_short_name} with guidance_scale={guidance_scale}"
echo "Model: {model_path}"
echo "Output: {output_dir}"
echo "Started at: $(date)"

python qwen_image_removetext_eval.py \\
    --model_path {model_path} \\
    --input_dir {base_config['input_dir']} \\
    --output_dir {output_dir} \\
    --prompt "{base_config['prompt']}" \\
    --num_inference_steps {base_config['num_inference_steps']} \\
    --guidance_scale {guidance_scale} \\
    --batch_size {base_config['batch_size']}

echo "Completed at: $(date)"
echo "Results saved to: {output_dir}"
"""
        
        # 保存单个脚本文件
        script_filename = f"eval_{model_short_name}_gs{guidance_scale}.sh"
        script_path = scripts_dir / script_filename
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 设置执行权限
        os.chmod(script_path, 0o755)
        
        all_scripts.append({
            'filename': script_filename,
            'path': str(script_path),
            'model_name': model_short_name,
            'model_path': model_path,
            'guidance_scale': guidance_scale,
            'output_dir': output_dir
        })
        
        print(f"Generated: {script_path}")
    
    # 生成批量执行脚本
    generate_batch_script(scripts_dir, all_scripts)
    
    # 生成并行执行脚本
    generate_parallel_script(scripts_dir, all_scripts)
    
    # 生成参数总结
    generate_summary(scripts_dir, all_scripts)
    
    print(f"\n总共生成了 {script_count} 个评估脚本")
    print(f"脚本保存在: {scripts_dir}")
    
    return all_scripts

def generate_batch_script(scripts_dir, all_scripts):
    """生成批量顺序执行脚本"""
    batch_content = """#!/bin/bash
# Batch execution script - runs all evaluations sequentially
# Generated automatically

echo "Starting batch evaluation at: $(date)"
echo "Total scripts to run: """ + str(len(all_scripts)) + """"

"""
    
    for i, script in enumerate(all_scripts, 1):
        batch_content += f"""
echo "\\n{'='*60}"
echo "Running script {i}/{len(all_scripts)}: {script['filename']}"
echo "Model: {script['model_name']}, Guidance Scale: {script['guidance_scale']}"
echo "{'='*60}"

./{script['filename']}

if [ $? -eq 0 ]; then
    echo "✓ {script['filename']} completed successfully"
else
    echo "✗ {script['filename']} failed"
    echo "Continuing with next script..."
fi
"""
    
    batch_content += """
echo "\\n==============================================" 
echo "Batch evaluation completed at: $(date)"
echo "=============================================="
"""
    
    batch_path = scripts_dir / "run_all_sequential.sh"
    with open(batch_path, 'w') as f:
        f.write(batch_content)
    os.chmod(batch_path, 0o755)
    
    print(f"Generated batch script: {batch_path}")

def generate_parallel_script(scripts_dir, all_scripts):
    """生成并行执行脚本"""
    parallel_content = """#!/bin/bash
# Parallel execution script - runs multiple evaluations simultaneously
# Generated automatically

echo "Starting parallel evaluation at: $(date)"
echo "Total scripts to run: """ + str(len(all_scripts)) + """"
echo "Running with 2 parallel processes (adjust as needed)"

# Function to run script and log results
run_script() {
    local script_name=$1
    local log_file="logs/${script_name%.sh}.log"
    
    echo "Starting $script_name at $(date)" > "$log_file"
    ./$script_name >> "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ $script_name completed successfully at $(date)" >> "$log_file"
        echo "✓ $script_name completed successfully"
    else
        echo "✗ $script_name failed at $(date)" >> "$log_file"
        echo "✗ $script_name failed"
    fi
}

# Create logs directory
mkdir -p logs

# Export function for parallel execution
export -f run_script

# List of all scripts
scripts=(
"""
    
    for script in all_scripts:
        parallel_content += f'    "{script["filename"]}"\n'
    
    parallel_content += """)

# Run scripts in parallel (2 at a time, adjust as needed)
printf '%s\\n' "${scripts[@]}" | xargs -n 1 -P 2 -I {} bash -c 'run_script "$@"' _ {}

echo "All parallel evaluations completed at: $(date)"
echo "Check individual log files in the 'logs' directory for details"
"""
    
    parallel_path = scripts_dir / "run_all_parallel.sh"
    with open(parallel_path, 'w') as f:
        f.write(parallel_content)
    os.chmod(parallel_path, 0o755)
    
    print(f"Generated parallel script: {parallel_path}")

def generate_summary(scripts_dir, all_scripts):
    """生成参数组合总结"""
    summary_content = """# Evaluation Scripts Summary

## Generated Scripts Overview

"""
    
    # 按模型分组
    models = {}
    for script in all_scripts:
        model = script['model_name']
        if model not in models:
            models[model] = []
        models[model].append(script)
    
    summary_content += f"**Total Scripts Generated:** {len(all_scripts)}\n\n"
    
    for model_name, scripts in models.items():
        summary_content += f"### {model_name.upper()} Model\n\n"
        summary_content += f"**Model Path:** `{scripts[0]['model_path']}`\n\n"
        summary_content += "| Script | Guidance Scale | Output Directory |\n"
        summary_content += "|--------|----------------|------------------|\n"
        
        for script in scripts:
            summary_content += f"| {script['filename']} | {script['guidance_scale']} | {script['output_dir']} |\n"
        
        summary_content += "\n"
    
    summary_content += """
## Usage Instructions

### Sequential Execution
```bash
cd eval_scripts
./run_all_sequential.sh
```

### Parallel Execution  
```bash
cd eval_scripts
./run_all_parallel.sh
```

### Individual Script Execution
```bash
cd eval_scripts
./eval_base_gs5.0.sh
```

## Directory Structure
```
eval_scripts/
├── eval_base_gs5.0.sh
├── eval_edit_gs5.0.sh
├── eval_edit2509_gs5.0.sh
├── run_all_sequential.sh
├── run_all_parallel.sh
├── logs/ (created during parallel execution)
└── README.md
```

## Parameters Used
- **Input Directory:** `/mnt/hero_blob/DATASET/SynthText/SCUT-EnsText_test`
- **Output Base Directory:** `/mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test`
- **Prompt:** "remove the text"
- **Inference Steps:** 50
- **Batch Size:** 1
- **Guidance Scale:** [5.0]
- **Models:** 3 variants (Base, Edit, Edit-2509)

## Output Directories
- Base Model: `/mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/base_gs5.0`
- Edit Model: `/mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/edit_gs5.0`
- Edit-2509 Model: `/mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/edit2509_gs5.0`

## Expected Results
Each evaluation will generate:
- Processed images in the output directory
- JSON files with evaluation metrics
- CSV files with per-image results
- FID and LOCAL-FID scores

## Monitoring Progress
- Sequential: Progress shown in terminal
- Parallel: Check `logs/` directory for individual script logs
"""
    
    summary_path = scripts_dir / "README.md"
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print(f"Generated summary: {summary_path}")

def generate_monitoring_script(scripts_dir):
    """生成监控脚本"""
    monitor_content = """#!/bin/bash
# Monitoring script to check evaluation progress
# Generated automatically

echo "Evaluation Progress Monitor"
echo "=========================="
echo "Checking at: $(date)"
echo ""

# Check if any scripts are running
running_processes=$(ps aux | grep "qwen_image_removetext_eval.py" | grep -v grep | wc -l)
echo "Currently running evaluations: $running_processes"

if [ $running_processes -gt 0 ]; then
    echo ""
    echo "Running processes:"
    ps aux | grep "qwen_image_removetext_eval.py" | grep -v grep | awk '{print $2, $11, $12, $13, $14, $15}'
fi

echo ""
echo "Output Directory Status:"
echo "========================"

# Check output directories
base_dirs=(
    "/mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/base_gs5.0"
    "/mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/edit_gs5.0"
    "/mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/edit2509_gs5.0"
)

for dir in "${base_dirs[@]}"; do
    if [ -d "$dir" ]; then
        file_count=$(find "$dir" -name "*.jpg" -o -name "*.png" | wc -l)
        json_count=$(find "$dir" -name "*.json" | wc -l)
        csv_count=$(find "$dir" -name "*.csv" | wc -l)
        echo "$(basename $dir): $file_count images, $json_count JSON files, $csv_count CSV files"
    else
        echo "$(basename $dir): Directory not found"
    fi
done

echo ""
echo "Log File Status (if running parallel):"
echo "======================================"

if [ -d "logs" ]; then
    for log_file in logs/*.log; do
        if [ -f "$log_file" ]; then
            echo "$(basename $log_file): $(wc -l < "$log_file") lines"
            # Show last few lines if file is being written to
            if [ -n "$(find "$log_file" -newermt '1 minute ago')" ]; then
                echo "  Recent activity detected - Last 3 lines:"
                tail -3 "$log_file" | sed 's/^/    /'
            fi
        fi
    done
else
    echo "No log directory found (not running in parallel mode)"
fi
"""
    
    monitor_path = scripts_dir / "monitor_progress.sh"
    with open(monitor_path, 'w') as f:
        f.write(monitor_content)
    os.chmod(monitor_path, 0o755)
    
    print(f"Generated monitoring script: {monitor_path}")

def main():
    """主函数"""
    print("🚀 Generating evaluation scripts...")
    print("=" * 60)
    
    scripts = generate_eval_scripts()
    
    # 生成监控脚本
    generate_monitoring_script(Path("eval_scripts"))
    
    print("\n" + "=" * 60)
    print("✅ Script generation completed!")
    print("\n📁 Generated files:")
    print("   - 3 individual evaluation scripts (one per model)")
    print("   - run_all_sequential.sh (sequential execution)")
    print("   - run_all_parallel.sh (parallel execution)")
    print("   - monitor_progress.sh (progress monitoring)")
    print("   - README.md (documentation)")
    
    print("\n🔧 Usage:")
    print("   cd eval_scripts")
    print("   ./run_all_sequential.sh      # Run all sequentially")
    print("   ./run_all_parallel.sh        # Run all in parallel")
    print("   ./monitor_progress.sh        # Monitor progress")
    print("   ./eval_base_gs5.0.sh         # Run individual script")
    
    print(f"\n📊 Total combinations: {len(scripts)}")
    print("   - 3 models × 1 guidance scale = 3 evaluations")
    
    print("\n🎯 Models to be evaluated:")
    for script in scripts:
        print(f"   - {script['model_name']}: {script['model_path']}")

if __name__ == "__main__":
    main()