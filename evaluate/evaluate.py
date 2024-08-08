import argparse
import json
import os
from openai import OpenAI

def process_data(base_url, api_key, model, max_tokens, input_file, output_dir):
    client = OpenAI(base_url=base_url, api_key=api_key)

    history = [{"role": "system", "content": """Character: As a Text Data Analyst specializing in heart failure, you're tasked with extracting key elements from medical texts to a JSON in IOB format.
Skills:
1. Name Extraction: Accurately identify and format names in IOB style.
2. heart failure Term Extraction: Extract essential heart failure related terms in IOB format. 
3. Structure Maintenance: Keep the JSON output structure intact and use placeholders for any missing data.
Constraints: Focus mainly on extracting names and heart failure related terms. Include all words from the sentence in the output JSON, even if some data is missing. Base your assumptions only on the provided text."""},
    ]

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for da in data:
        history.append({"role": "user", "content": da["instruction"]})
        completion = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.1,
            response_format={"type": "json_object"},
            stream=False,
            max_tokens=max_tokens,
        )
        resp = (completion.choices[0].message.content)
        print(resp)
        try:
            resp = json.loads(resp)
        except:
            resp = json.loads(f'{"terms":"{resp}"}')
        print(resp)

        file_path = os.path.join(output_dir, f'{da["language"]}_{da["dataset"]}_resp.json')

        if os.path.exists(file_path):
            with open(file_path, 'a', encoding='utf-8') as f:
                json.dump({"instruction": da["instruction"], "output": resp["terms"], "answer": json.loads(da["output"])["terms"]}, f, ensure_ascii=False)
                f.write('\n')
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({"instruction": da["instruction"], "output": resp["terms"], "answer": json.loads(da["output"])["terms"]}, f, ensure_ascii=False)
                f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process medical text data.')
    parser.add_argument('--base_url', type=str, default="http://localhost:11434/v1", help='Base URL for the OpenAI API')
    parser.add_argument('--api_key', type=str, default="your_api_key", help='API key for the OpenAI API')
    parser.add_argument('--model', type=str, default="llama3_term", help='Model to use for the OpenAI API')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum number of tokens for the response')
    parser.add_argument('--input_file', type=str, default="test.json", help='Path to the input JSON file')
    parser.add_argument('--output_dir', type=str, default="result", help='Directory to save the output JSON files')

    args = parser.parse_args()
    process_data(args.base_url, args.api_key, args.model, args.max_tokens, args.input_file, args.output_dir)