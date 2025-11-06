#!/bin/bash
# Generated automatically
export CUDA_VISIBLE_DEVICES=0
echo "Starting evaluation for base with guidance_scale=5.0"
echo "Model: /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509"
echo "Output: /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_iprompt_nomask_remove_SCUT-EnsText_test"
echo "Started at: $(date)"

python qwen_image_ipnomask_eval.py \
    --model_path /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509 \
    --input_dir /mnt/hero_blob/DATASET/SynthText/SCUT-EnsText_test \
    --output_dir /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_iprompt_nomask_remove_SCUT-EnsText_test \
    --prompt "remove all the text" \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \

echo "Completed at: $(date)"
echo "Results saved to: /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_iprompt_nomask_remove_SCUT-EnsText_test"
