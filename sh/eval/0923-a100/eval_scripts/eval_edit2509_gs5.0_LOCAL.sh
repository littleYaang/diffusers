#!/bin/bash
# Evaluation script for base with guidance_scale=5.0
# Generated automatically
export CUDA_VISIBLE_DEVICES=0
echo "Starting evaluation for base with guidance_scale=5.0"
echo "Model: /home/v-qinhyang/code/hero_blob/preweight/Qwen/Qwen-Image"
echo "Output: /home/v-qinhyang/code/hero_blob/TESTRESULTS/SCUT-EnsText_test/Qwen-Image-Edit-all"
echo "Started at: $(date)"

python qwen_image_removetext_eval.py \
    --model_path /home/v-qinhyang/code/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509 \
    --input_dir /home/v-qinhyang/code/hero_blob/DATASET/SynthText/SCUT-EnsText_test \
    --output_dir /home/v-qinhyang/code/hero_blob/TESTRESULTS/SCUT-EnsText_test/Qwen-Image-Edit-all \
    --prompt "remove the text" \
    --num_inference_steps 30 \
    --guidance_scale 5.0 \

echo "Completed at: $(date)"
echo "Results saved to: /home/v-qinhyang/code/hero_blob/TESTRESULTS/SCUT-EnsText_test/Qwen-Image-Edit-all"
