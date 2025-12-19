
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from unsloth import FastLanguageModel
from peft import (
    LoraConfig,
    AdaLoraConfig,
    PrefixTuningConfig,
    AdaptionPromptConfig,
    IA3Config,
    get_peft_model as hf_get_peft_model,
)

class FT_Models:
    def __init__(self, model_spec, logger=None):
        self.model_spec = model_spec
        self.logger = logger

        self.models = {
            "Q1.5B": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
            "Q7B": "unsloth/DeepSeek-R1-Distill-Qwen-7B",
            "Llama": "unsloth/meta-llama-3.1-8b-instruct",
            "Q14B": "unsloth/DeepSeek-R1-Distill-Qwen-14B",
            "Qwen": "unsloth/Qwen2-7B-Instruct"
        }

    def get_tokenizer(self, model_name):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.models[model_name],
            max_seq_length = 1024,
            load_in_4bit = False,
        )

        return tokenizer

    def get_zs_model(self, args):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.models[args.model],
            max_seq_length = args.max_seq_length,
            load_in_4bit = args.load_4bit == 1,
        )
        FastLanguageModel.for_inference(model)

        return model, tokenizer

    def get_ft_model(self, args):
        # Load base model with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.models[args.model],
            max_seq_length = args.max_seq_length,
            load_in_4bit = args.load_4bit == 1,
            device_map=None,
        )
        model.to("cuda")

        # ------------------------------------------------------
        # 🔥 disable Flash Attention & fused kernels
        # ------------------------------------------------------
        # model.config.use_flash_attn = False
        # model.config.use_fused_sdpa = False
        # model.config.use_xformers = False
        # model.config.attn_implementation = "eager"

        # Default to lora if argument is missing
        peft_method = getattr(args, "peft_method", "lora").lower()

        # Same target modules you were using before
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        # =============================
        # CHOOSE PEFT METHOD
        # =============================

        if peft_method in ["lora", "qlora"]:
            # LoRA / QLoRA using Unsloth's optimized helper
            # QLoRA is enabled by setting --load_4bit 1
            model = FastLanguageModel.get_peft_model(
                model,
                r = args.rank,
                target_modules = target_modules,
                lora_alpha = 16,
                lora_dropout = 0,
                bias = "none",
                use_gradient_checkpointing = "unsloth",
                random_state = 42,
                use_rslora = False,
                loftq_config = None,
            )

        elif peft_method == "adalora":
            # AdaLoRA using standard PEFT
            cfg = AdaLoraConfig(
                init_r = args.rank,
                lora_alpha = 16,
                target_modules = target_modules,
                task_type = "CAUSAL_LM",
                total_step = args.max_steps,
            )
            model = hf_get_peft_model(model, cfg)

        elif peft_method == "adapters":
            # AdaptionPrompt (adapter-style PEFT)
            cfg = AdaptionPromptConfig(
                task_type = "CAUSAL_LM",
                adapter_layers = args.rank,
            )
            model = hf_get_peft_model(model, cfg)

        elif peft_method == "prefix":
            # Prefix tuning
            cfg = PrefixTuningConfig(
                task_type = "CAUSAL_LM",
                num_virtual_tokens = 30,
            )
            model = hf_get_peft_model(model, cfg)

        elif peft_method == "ia3":
            ia3_config = IA3Config(
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                feedforward_modules=["gate_proj", "up_proj", "down_proj"],
            )

            model = hf_get_peft_model(model, ia3_config)
            model.print_trainable_parameters()

        else:
            raise ValueError(f"Unknown peft_method: {peft_method}")

        if self.logger:
            self.logger(f"Using PEFT method: {peft_method}\n")

        return model, tokenizer

        # try:
        #     model = FastLanguageModel.get_peft_model(
        #         model,
        #         r = 4,
        #         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        #         lora_alpha = 16,
        #         lora_dropout = 0,
        #         bias = "none",
        #         use_gradient_checkpointing = "unsloth",
        #         random_state = 42,
        #         use_rslora = False,
        #         loftq_config = None,
        #     )

        #     if self.logger is not None:
        #         self.logger("LoRA on q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj\n\n")
        # except:
        #     model = FastLanguageModel.get_peft_model(
        #         model,
        #         r = 4,
        #         target_modules=["q_proj", "k_proj", "v_proj"],
        #         lora_alpha = 16,
        #         lora_dropout = 0,
        #         bias = "none",
        #         use_gradient_checkpointing = "unsloth",
        #         random_state = 42,
        #         use_rslora = False,
        #         loftq_config = None,
        #     )

        #     self.logger("LoRA on q_proj, k_proj, v_proj\n\n")

        # return model, tokenizer