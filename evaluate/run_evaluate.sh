#!/bin/bash

# Default values for the arguments
MODEL="weiiv/llama3.1_term_json"
MAX_TOKENS=4096
INPUT_FILE="test.json"
OUTPUT_DIR="results"
SYSTEM_PROMPT="Extract terms from the following data and format it in JSON using IOB notation."
model_type="llama3"
curl -o test.json https://huggingface.co/datasets/weiiv/terms_new/raw/main/test.json
mkdir $OUTPUT_DIR
# Run the Python script with the provided arguments
python3 eva.py \
    --model "$MODEL" \
    --max_tokens "$MAX_TOKENS" \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$model_type" \
    --system_prompt "$SYSTEM_PROMPT"
