#set CUDA_LAUNCH_BLOCKING=1
# python zs_inference.py --task irab --model Q14B --prompt_lang ar --load_4bit 1 --save_path ./zs_preds
# python zs_inference.py --shots 3 --task irab --model Q14B --prompt_lang ar --save_path ./3s_preds
python zs_inference.py --shots 5 --task irab --model Q14B --prompt_lang ar --save_path ./5s_preds