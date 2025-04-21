import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configs
BASE_MODEL_NAME = "./Llama-3.1-8B-Instruct"
REWARD_MODEL_NAME = "./Llama-3.1-8B-Instruct"  # same model acting as judge
DATA_PATH = "cleaned_qna.json"  # format: {"prompt": "...", "output": "..."}
OUTPUT_PATH = "dpo_ready_data.json"
MAX_NEW_TOKENS = 32
TEMPERATURE = 1.0

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading tokenizer and base model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
reward_model = AutoModelForCausalLM.from_pretrained(
    REWARD_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME, trust_remote_code=True)

def score_answer(prompt, answer):
    # Simplified scoring: logprob of answer continuation
    input_text = f"[INST] {prompt} [/INST] {answer}"
    inputs = reward_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(base_model.device)
    with torch.no_grad():
        outputs = reward_model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        avg_log_prob = selected_log_probs.mean().item()
    return avg_log_prob

def generate_answer(prompt, context):
    input_text = f"[INST] You are a helpful assistant. Based **only** on the context, answer the question. Do **not** repeat the question or context.\n\nContext:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(input_text):].strip()

# Process dataset
with open(DATA_PATH, "r") as f:
    data = json.load(f)

print("Generating new completions and comparing to dataset answers...", flush=True)
with open(OUTPUT_PATH, "w") as out_f:
    for obj in tqdm(data):
        prompt = obj["question"]
        original_answer = obj["answer"]
        context = obj["citation"]

        # Generate alternative answer
        generated_answer = generate_answer(prompt, context)
        # Score both
        original_score = score_answer(prompt, original_answer)
        generated_score = score_answer(prompt, generated_answer)

        if original_score >= generated_score:
            chosen, rejected = original_answer, generated_answer
        else:
            chosen, rejected = generated_answer, original_answer

        out_obj = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
        out_f.write(json.dumps(out_obj) + "\n")

print(f"Saved DPO-ready dataset to {OUTPUT_PATH}", flush=True)
