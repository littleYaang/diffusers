#!/bin/bash
# Generated automatically
export CUDA_VISIBLE_DEVICES=2
echo "Starting evaluation for base with guidance_scale=5.0"
echo "Model: /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509"
echo "Output: /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_rectmask_remove_ours_tiny_text"
echo "Started at: $(date)"

python qwen_image_removetext_eval.py \
    --model_path /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509 \
    --input_dir /mnt/hero_blob/DATASET/SynthText/AnyTE_bench/AnyTE-all/tiny_text \
    --output_dir  /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_rectmask_remove_ours_tiny_text\
    --prompt "remove all the text" \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \

echo "Completed at: $(date)"
echo "Results saved to: /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_rectmask_remove_ours_tiny_text"
