---
creation date: 2024-07-19 20:24
tags: 
aliases:
---

### Environment Setup
```
pip install openai
curl -fsSL https://ollama.com/install.sh | sh && ollama serve
```

### Creating an Ollama Model
You can find modelfile templates at https://ollama.com/library. Copy the appropriate template and parameters for different models.

Alternatively, you can run a base model first, such as `ollama run llama3`, then use `ollama show llama3 --modelfile` and copy the output to a new file.

Copy the address of your trained GGUF file after the FROM statement, save the file, and run:
`ollama create llama3term -f modelfile_location`

Here's an example:
**change GGUF_location**
```
FROM GGUF_location

TEMPLATE "{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"
PARAMETER num_keep 24
PARAMETER stop <|start_header_id|>
PARAMETER stop <|end_header_id|>
PARAMETER stop <|eot_id|>
```

Then run the current model:
`ollama run llama3term`

Download the test dataset:
`curl -o test.json https://huggingface.co/datasets/weiiv/terms_new/raw/main/test.json`

Modify the parameters in `run_evaluate.sh`, create a folder for output results, add execution permissions with `chmod +x run_evaluate.sh`, and then run it.
