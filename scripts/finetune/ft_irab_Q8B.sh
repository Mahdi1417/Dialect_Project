CUDA_VISIBLE_DEVICES=0 python finetune.py --task irab --model Q8B --prompt_lang ar --load_4bit 1
# CUDA_VISIBLE_DEVICES=0 python ft_inference.py --task irab --model Q14B --prompt_lang ar --load_4bit
# CUDA_VISIBLE_DEVICES=0 python ft_eval.py --task irab --model Q14B --prompt_lang ar --load_4bit