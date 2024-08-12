#!/bin/bash



mkdir results_llama3
python eva.py --model "../train/Meta-Llama-3.1-8B-prompt" --max_tokens 4096 --input_file "test.json" --output_dir "results_llama3" --model_type "llama3"
