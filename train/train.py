# Import libraries
import argparse
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Define formatting function
def formatting_prompts_func(examples, alpaca_prompt, system_prompt,tokenizer):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    EOS_TOKEN = tokenizer.eos_token 
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(system_prompt, instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

# Load and format dataset
def load_and_format_dataset(dataset_name, split, prompt_format, system_prompt,tokenizer):
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.map(lambda examples: formatting_prompts_func(examples, prompt_format, system_prompt,tokenizer), batched=True)
    return dataset

# Print the first sample
def print_first_sample(dataset):
    print(dataset[0])

# Define training arguments
def get_training_args(batch_size, num_epochs, learning_rate, gradient_accumulation_steps, warmup_steps, logging_steps, optim, weight_decay, lr_scheduler_type, seed, output_dir):
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=logging_steps,
        optim=optim,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        output_dir=output_dir,
    )
    return training_args

# Initialize trainer
def initialize_trainer(model, tokenizer, dataset, training_args, max_seq_length):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=training_args,
    )
    return trainer

# Train the model
def train_model(trainer):
    trainer_stats = trainer.train()
    return trainer_stats

# Save the model
def save_model(model, tokenizer, model_dir, hub_model_id, hub_token, quantization_method,save_lora,save_16bit,push_lora,push_gguf):
    if save_lora:
        model.save_pretrained_merged(model_dir, tokenizer, save_method="lora")
        model.save_pretrained("lora_model") # Local saving
        tokenizer.save_pretrained("lora_model")
    if save_16bit:
        model.save_pretrained_merged(model_dir, tokenizer, save_method = "merged_16bit")
    if push_lora:
        model.push_to_hub_merged(hub_model_id, tokenizer, save_method="lora", token=hub_token)
    if push_gguf:
        model.push_to_hub_gguf(hub_model_id, tokenizer, quantization_method=quantization_method, token=hub_token)
def evaluate_model(model, tokenizer):
    #  alpaca_prompt = Copied from above
    import json, os
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    #load json data
    with open('test.json','r',encoding='utf-8') as f:
        data = json.load(f)

    for da in data:
        inputs = tokenizer(
        [
            alpaca_prompt.format(
                da["instruction"], # instruction
                "", # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)

        resp = tokenizer.batch_decode(outputs,skip_special_tokens=True)
        print(resp)
        try:
            resp = json.loads(resp)
        except:
            resp = {"terms":resp}
        # print(resp["terms"])
        # print(json.loads(da["output"])["terms"])

        file_path = '/content/drive/MyDrive/results/'+da["language"]+'_'+da["dataset"]+'_resp.json'
        try:
            if os.path.exists(file_path):
                with open(file_path, 'a', encoding='utf-8') as f:
                    json.dump({"instruction":da["instruction"], "output":resp["terms"], "answer":json.loads(da["output"])["terms"]}, f, ensure_ascii=False)
                    f.write('\n')
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({"instruction":da["instruction"], "output":resp["terms"], "answer":json.loads(da["output"])["terms"]}, f, ensure_ascii=False)
                    f.write('\n')
        except:
            print("Error in writing to file")
            print(da)

def get_prompt_format(model_name):
    if model_name == "phi3":
        return r"""<s><|system|>{}<|end|>
    <|user|>{}<|end|>
    <|assistant|>{}"""
    elif model_name == "llama3":
        return r"""<|begin_of_text|><|start_header_id|>system<|end_header_id><|eot_id|>
{}
<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}"""
    elif model_name == "gemma":
        return r"""<start_of_turn>user
    {}"{}" <end_of_turn>
    <start_of_turn>model
    {}"""
    elif model_name == "qwen2":
        return r"""<|im_start|>system
    {}<|im_end|>
    <|im_start|>user
    {}<|im_end|>
    <|im_start|>assistant<|eot_id|>
    {}"""
# Main function
def main():
    parser = argparse.ArgumentParser(description="Train and save model")
    parser.add_argument("--batch_size", type=int, default=3, help="Training batch size per device")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="unsloth/Phi-3-mini-4k-instruct-bnb-4bit", help="Model name")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--dtype", type=str, default=None, help="Data type")
    parser.add_argument("--load_in_4bit", type=bool, default=True, help="Load in 4-bit")
    parser.add_argument("--r", type=int, default=16, help="r parameter")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0, help="lora dropout parameter")
    parser.add_argument("--bias", type=str, default="none", help="bias parameter")
#     parser.add_argument("--prompt_format", type=str, default="""<s><|system|>{}<|end|>
# <|user|>{}<|end|>
# <|assistant|>{}<|end|>""", help="alpaca prompt")
    parser.add_argument("--system_prompt", type=str, default="""Character: As a Text Data Analyst, you're tasked with extracting key elements from text to a JSON in IOB format.
Skills:
Name Extraction: Accurately identify and format names in IOB style.
Domain Term Extraction: Extract essential domain terms in IOB format.
Structure Maintenance: Keep the JSON output structure intact and use placeholders for any missing data.
Constraints: Focus mainly on extracting names and keywords. Include all words from the sentence in the output JSON, even if some data is missing. Base your assumptions only on the provided text.
""", help="system prompt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=5, help="Logging steps")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
    parser.add_argument("--seed", type=int, default=123, help="Training Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--model_dir", type=str, default="model", help="Model save directory")
    parser.add_argument("--hub_model_id", type=str, default="weiiv/terms", help="Hub model ID")
    parser.add_argument("--hub_token", type=str, default=None, help="Hub token")
    parser.add_argument("--quantization_method", type=str, default="q4_k_m", help="Quantization method")
    parser.add_argument("--save_lora", type=bool, default=True, help="Save LoRA")
    parser.add_argument("--save_16bit", type=bool, default=False, help="Save 16bit")
    parser.add_argument("--push_lora", type=bool, default=False, help="Push LoRA")
    parser.add_argument("--push_gguf", type=bool, default=False, help="Push GGUF")
    parser.add_argument("--model_type",type=str,default="llama3")
    args = parser.parse_args()
    prompt_format = get_prompt_format(args.model_type)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias, 
        use_gradient_checkpointing="unsloth", 
        random_state=args.seed,
    )
    dataset = load_and_format_dataset("weiiv/terms_new", "train", prompt_format, args.system_prompt,tokenizer)
    print_first_sample(dataset)
    training_args = get_training_args(args.batch_size, args.num_epochs, args.learning_rate, args.gradient_accumulation_steps, args.warmup_steps, args.logging_steps, args.optim, args.weight_decay, args.lr_scheduler_type, args.seed, args.output_dir)
    trainer = initialize_trainer(model, tokenizer, dataset, training_args, args.max_seq_length)
    train_model(trainer)
    save_model(model, tokenizer, args.model_dir, args.hub_model_id, args.hub_token, args.quantization_method,args.save_lora,args.save_16bit,args.push_lora,args.push_gguf)

if __name__ == "__main__":
    main()
