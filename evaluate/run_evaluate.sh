#!/bin/bash

# Default values for the arguments
MODEL="weiiv/llama3.1_term_json"
MAX_TOKENS=4096
INPUT_FILE="test.json"
OUTPUT_DIR="results"
PROMPT_FORMAT="<|begin_of_text|><|start_header_id|>system<|end_header_id><|eot_id|>\n{}\n<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}"
SYSTEM_PROMPT="Extract terms from the following data and format it in JSON using IOB notation."

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --max_tokens) MAX_TOKENS="$2"; shift ;;
        --input_file) INPUT_FILE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --prompt_format) PROMPT_FORMAT="$2"; shift ;;
        --system_prompt) SYSTEM_PROMPT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
curl -o test.json https://huggingface.co/datasets/weiiv/terms_new/raw/main/test.json
mkdir $OUTPUT_DIR
# Run the Python script with the provided arguments
python3 eva.py \
    --model "$MODEL" \
    --max_tokens "$MAX_TOKENS" \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --prompt_format "$PROMPT_FORMAT" \
    --system_prompt "$SYSTEM_PROMPT"