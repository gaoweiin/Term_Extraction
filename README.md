# term-extraction
```
.gitignore
evaluate/
    evaluate.py
    README.md
    run_evaluate.sh
README.md
train/
    README.md
    train.py
    train.sh
```

prompt format:
gemma
"""
<start_of_turn>user
{}"{}" <end_of_turn>
<start_of_turn>model
{}
"""

phi3
"""
<s><|system|>{}<|end|>
<|user|>{}<|end|>
<|assistant|>{}
"""

qwen2
"""
<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant<|eot_id|>
{}
"""

llama3 
"""<|begin_of_text|><|start_header_id|>system<|end_header_id><|eot_id|>
{}
<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}"""
