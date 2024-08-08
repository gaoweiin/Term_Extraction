#!/bin/bash

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id><|eot_id|>\
{}\
<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}"

python train.py --model_name "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --num_epochs 1 --prompt_format $prompt --batch_size 8 --system_prompt "Extract terms from the following data and format it in JSON using IOB notation."  --hub_model_id "weiiv/llama3.1_term_json" --hub_token " " --push_lora 1
