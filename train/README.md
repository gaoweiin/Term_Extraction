---
creation date: 2024-07-17 16:44
tags: 
aliases:
---
### Training

#### Command-Line Arguments

- `--batch_size`: Training batch size per device (default: 3)
- `--num_epochs`: Number of training epochs (default: 1)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--model_name`: Model name (default: "unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
- `--max_seq_length`: Maximum sequence length (default: 4096)
- `--dtype`: Data type (default: None)
- `--load_in_4bit`: Load in 4-bit (default: True)
- `--r`: r parameter (default: 16)
- `--lora_alpha`: lora alpha parameter (default: 16)
- `--lora_dropout`: lora dropout parameter (default: 0)
- `--bias`: bias parameter (default: "none")
- `--prompt_format`: alpaca prompt format (default: specified in the script)
- `--system_prompt`: system prompt (default: specified in the script)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `--warmup_steps`: Warmup steps (default: 5)
- `--logging_steps`: Logging steps (default: 5)
- `--optim`: Optimizer (default: "adamw_8bit")
- `--weight_decay`: Weight decay (default: 0.01)
- `--lr_scheduler_type`: Learning rate scheduler type (default: "linear")
- `--seed`: Training random seed (default: 123)
- `--output_dir`: Output directory (default: "outputs")
- `--model_dir`: Model save directory (default: "model")
- `--hub_model_id`: Hub model ID (default: "weiiv/terms")
- `--hub_token`: Hub token (default: None)
- `--quantization_method`: Quantization method (default: "q4_k_m")
- `--save_lora`: Save LoRA (default: True)
- `--push_lora`: Push LoRA (default: False)
- `--push_gguf`: Push GGUF (default: False)

#### Example Usage

```bash
#!/bin/bash

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id><|eot_id|>\
{}\
<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}"

python ttt.py --model_name "unsloth/llama-3-8b-Instruct-bnb-4bit" --system_prompt $prompt
```


