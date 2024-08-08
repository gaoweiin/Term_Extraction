from unsloth import FastLanguageModel
import json, os
import argparse
def process_data(model_name,max_tokens,input_file,output_dir,prompt_format,system_prompt):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_tokens,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) 

    with open(input_file,'r',encoding='utf-8') as f:
        data = json.load(f)

    for da in data:
        inputs = tokenizer(
        [
            prompt_format.format(
                system_prompt,
                da["instruction"], 
                "",
            )
        ], return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = max_tokens, use_cache = True, return_dict_in_generate=True)
        generated_tokens = outputs.sequences[:, inputs['input_ids'].shape[-1]:]

        resp = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(resp)
        try:
            resp = json.loads(resp[0])
        except:
            resp = {"terms":resp}
        # print(resp["terms"])
        # print(json.loads(da["output"])["terms"])

        file_path = output_dir+da["language"]+'_'+da["dataset"]+'_resp.json'
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process medical text data.')
    parser.add_argument('--model', type=str, default="weiiv/llama3.1_term_json", help='Model')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum number of tokens for the response')
    parser.add_argument('--input_file', type=str, default="test.json", help='Path to the input JSON file')
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save the output JSON files')
    parser.add_argument("--prompt_format", type=str, default="""<|begin_of_text|><|start_header_id|>system<|end_header_id><|eot_id|>
{}
<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}""", help="prompt format")
    parser.add_argument("--system_prompt", type=str, default="Extract terms from the following data and format it in JSON using IOB notation.", help="system prompt")
    args = parser.parse_args()
    process_data(args.model, args.max_tokens, args.input_file, args.output_dir, args.prompt_format, args.system_prompt)