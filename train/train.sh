#!/bin/bash


python train.py --model_name "unsloth/gemma-2-9b-it-bnb-4bit" --num_epochs 1 --model_type "gemma" --batch_size 8 --system_prompt "Extract terms from the following data and format it in JSON using IOB notation."  --model_dir "gemma-2-9b-it-bnb-4bit"

python train.py --model_name "unsloth/Phi-3-medium-4k-instruct-bnb-4bit" --num_epochs 1 --model_type "phi3" --batch_size 8 --system_prompt "Extract terms from the following data and format it in JSON using IOB notation."  --model_dir "Phi-3-medium-4k-instruct-bnb-4bit"

python train.py --model_name "unsloth/Qwen2-7B-Instruct-bnb-4bit" --num_epochs 1 --model_type "qwen2" --batch_size 8 --system_prompt "Extract terms from the following data and format it in JSON using IOB notation." --model_dir "Qwen2-7B-Instruct-bnb-4bit"

python train.py --model_name "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --num_epochs 1 --model_type "llama3 --batch_size 8 --model_dir "Meta-Llama-3.1-8B-prompt"
