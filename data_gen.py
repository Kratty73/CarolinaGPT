from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import json
from typing import List, Dict
import requests

# Set local model path
local_model_path = "./Llama-3.1-8B-Instruct/"  # update as needed

# Load tokenizer from local path
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Load model in fp16 (no quantization needed)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Create a text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)
responses = ""
def load_json_files(json_paths: List[str]) -> List[Dict]:
    """Load multiple JSON files and return a list of data entries."""
    combined_data = []
    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                combined_data.extend(data)
            else:
                combined_data.append(data)
    return combined_data

def construct_prompt(entry: Dict, index: int) -> str:
    context = json.dumps(entry, indent=2)

    return f"""You are a helpful assistant tasked with reading JSON data and generating a generic question, answer, and citation based on it.

JSON Data:
{context}

Instructions:
1. Create multiple(0 to 5 max) thoughtful and unique question based strictly on the content above.
2. Each question should be interpretable on its own, without needing to reference the JSON structure directly.
3. Some sections might not have enough information to generate a question. Do not generate a question in this case.
4. Provide an accurate answer using only the information present in the JSON.
5. For the citation, copy **only the exact sentence(s) or text snippets** from the JSON that support the answer — do not summarize or paraphrase. Use only **verbatim text** from the JSON content.
6. Each line with question must start with Question do not break this instruction.
7. Each line with answer must start with Answer do not break this instruction.
8. Each line with citation must start with Citation do not break this instruction.

Format:
Question: <question here>
Answer: <answer here>
Citation: <copied sentence(s) from JSON that support the answer — no changes>
"""

def generate_question_answer(entry: Dict, index: int, verbose: bool = True) -> Dict:
    """Generate a question-answer-citation from a JSON entry, with dynamic console output."""
    global responses
    prompt = construct_prompt(entry, index)

    if verbose:
        print(f"\n🟡 Generating Q&A for entry #{index + 1}...", flush=True)

    # Generate content from Llama
    response = generator(prompt, max_new_tokens=300, temperature=0.7)[0]["generated_text"][len(prompt):].strip()

    responses += response

    if verbose:
        print(f"✅ Raw response received for entry #{index + 1}:\n", flush=True)
        # print(response)
        print("-" * 80, flush=True)

    return parse_response(response)

def parse_response(response_text: str) -> Dict:
    lines = response_text.strip().splitlines()
    question, answer, citation = "", "", ""
    response = []

    for line in lines:
        if line.lower().startswith("question:"):
            question = line.split(":", 1)[1].strip()
        elif line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
        elif line.lower().startswith("citation:"):
            citation = line.split(":", 1)[1].strip()
            response.append({"question": question, "answer": answer, "citation": citation})
        else:
            question, answer, citation = "", "", ""

    return response


def generate_qna(json_paths: List[str], output_path: str = "generated_questions_2.json", output_responses: str = "generated_responses_2.txt"):
    data_entries = load_json_files(json_paths)
    global responses
    for _, grouped_entry in enumerate(data_entries):
        generated_qas = []
        responses = ""
        for idx, entry in enumerate(grouped_entry['content']):
            try:
                qas = generate_question_answer(entry, idx)
                for qa in qas:
                    if qa["question"] and qa["answer"]:
                        generated_qas.append(qa)
            except Exception as e:
                print(f"[Error] Entry {idx}: {e}", flush=True)

        with open(output_path, 'a', encoding='utf-8') as out_file:
            json.dump(generated_qas, out_file, indent=2)
            print(f"Saved {len(generated_qas)} Q&A pairs to {output_path}", flush=True)
        with open(output_responses, 'a', encoding='utf-8') as out_file:
            out_file.write(responses)
    return

json_files = [f"./data/unc_cs_program_cleaned_batch_{batch_id}.json" for batch_id in range(1,9)]
generate_qna(json_files)