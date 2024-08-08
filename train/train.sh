#!/bin/bash

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id><|eot_id|>\
{}\
<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}"

gemma="<start_of_turn>user\
{}"{}" <end_of_turn>\
<start_of_turn>model\
{}"

phi3="<s><|system|>{}<|end|>\
<|user|>{}<|end|>\
<|assistant|>{}"

qwen2="<|im_start|>system\
{}<|im_end|>\
<|im_start|>user\
{}<|im_end|>\
<|im_start|>assistant<|eot_id|>\
{}"

system_prompt_1 = "Character: As a Text Data Analyst specializing in heart failure, you're tasked with extracting key elements from medical texts to a JSON in IOB format.\
Skills:\
1. Name Extraction: Accurately identify and format names in IOB style.\
2. Heart failure Term Extraction: Extract essential heart failure-related terms in IOB format.\
3. Structure Maintenance: Keep the JSON output structure intact and use placeholders for any missing data.\
Constraints: Focus mainly on extracting names and heart failure-related terms. Include all words from the sentence in the output JSON, even if some data is missing. Base your assumptions only on the provided text."

python train.py --model_name "unsloth/gemma-2-9b-it-bnb-4bit" --num_epochs 1 --prompt_format $gemma --batch_size 8 --system_prompt "Extract terms from the following data and format it in JSON using IOB notation."  --model_dir "gemma-2-9b-it-bnb-4bit"

python train.py --model_name "unsloth/Phi-3-medium-4k-instruct-bnb-4bit" --num_epochs 1 --prompt_format $phi3 --batch_size 8 --system_prompt "Extract terms from the following data and format it in JSON using IOB notation."  --model_dir "Phi-3-medium-4k-instruct-bnb-4bit"

python train.py --model_name "unsloth/Qwen2-7B-Instruct-bnb-4bit" --num_epochs 1 --prompt_format $qwen2 --batch_size 8 --system_prompt "Extract terms from the following data and format it in JSON using IOB notation." --model_dir "Qwen2-7B-Instruct-bnb-4bit"

python train.py --model_name "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --num_epochs 1 --prompt_format $prompt --batch_size 8 --system_prompt $system_prompt_1 --model_dir "Meta-Llama-3.1-8B-prompt"
