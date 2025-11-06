python qwen_image_removetext_eval.py \
    --model_path /mnt/hero_blob/preweight/Qwen/Qwen-Image \
    --input_dir /mnt/hero_blob/DATASET/SynthText/SCUT-EnsText_test \
    --output_dir /mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/all \
    --prompt "remove the text" \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --batch_size 1