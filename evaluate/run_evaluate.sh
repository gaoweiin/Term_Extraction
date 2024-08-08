#!/bin/bash



system_prompt_1 = "Character: As a Text Data Analyst specializing in heart failure, you're tasked with extracting key elements from medical texts to a JSON in IOB format.\
Skills:\
1. Name Extraction: Accurately identify and format names in IOB style.\
2. Heart failure Term Extraction: Extract essential heart failure-related terms in IOB format.\
3. Structure Maintenance: Keep the JSON output structure intact and use placeholders for any missing data.\
Constraints: Focus mainly on extracting names and heart failure-related terms. Include all words from the sentence in the output JSON, even if some data is missing. Base your assumptions only on the provided text."



curl -o test.json https://huggingface.co/datasets/weiiv/terms_new/raw/main/test.json

mkdir results_gemma
python eva.py --model "gemma-2-9b-it-bnb-4bit" --max_tokens 4096 --input_file "test.json" --output_dir "results_gemma" --model_type "gemma" --system_prompt "Extract terms from the following data and format it in JSON using IOB notation."
mkdir results_phi3
python eva.py --model "Phi-3-medium-4k-instruct-bnb-4bit" --max_tokens 4096 --input_file "test.json" --output_dir "results_phi3" --model_type "phi3" --system_prompt "Extract terms from the following data and format it in JSON using IOB notation."
mkdir results_qwen2
python eva.py --model "Qwen2-7B-Instruct-bnb-4bit" --max_tokens 4096 --input_file "test.json" --output_dir "results_qwen2" --model_type "qwen2" --system_prompt "Extract terms from the following data and format it in JSON using IOB notation."
mkdir results_llama3
python eva.py --model "Meta-Llama-3.1-8B-prompt" --max_tokens 4096 --input_file "test.json" --output_dir "results_llama3" --model_type "llama3" --system_prompt $system_prompt_1
