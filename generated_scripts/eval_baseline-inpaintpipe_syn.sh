#!/bin/bash
# Generated automatically - Model baseline-inpaintpipe on syn
export CUDA_VISIBLE_DEVICES=1
echo "Starting evaluation for model baseline-inpaintpipe on syn with guidance_scale=5.0"
echo "Model: /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509"
echo "LoRA: /home/v-qinhyang/code/hero_blob/qwenoutput/plain_lora/edit_plus_scut2749_lora_128d_5e-5_16ep-000004.safetensors"
echo "Input: /mnt/hero_blob/DATASET/SynthText/Syn-Text/syn_test"
echo "Output: /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_ablation_syn_lora128_baseline-inpaintpipe_ens_syn"
echo "Started at: $(date)"

python inference_qwen_image_edit_our.py \
    --model_path /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509 \
    --input_dir /mnt/hero_blob/DATASET/SynthText/Syn-Text/syn_test \
    --output_dir /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_ablation_syn_lora128_baseline-inpaintpipe_ens_syn \
    --inpaint_pipe True \
    --num_inference_steps 50 \
    --guidance_scale 5.0

echo "Completed at: $(date)"
echo "Results saved to: /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_ablation_syn_lora128_baseline-inpaintpipe_ens_syn"
