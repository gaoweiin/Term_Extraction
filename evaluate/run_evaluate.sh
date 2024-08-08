#!/bin/bash

base_url="http://localhost:11434/v1"
api_key="your_api_key"
model="llama3_term"
max_tokens=2048
input_file="test.json"
output_dir="result"

python3 evaluate.py --base_url "$base_url" --api_key "$api_key" --model "$model" --max_tokens "$max_tokens" --input_file "$input_file" --output_dir "$output_dir"