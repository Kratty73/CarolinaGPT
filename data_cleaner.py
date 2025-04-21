import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./Llama-3.1-8B-Instruct"
INPUT_FILE = "qna_filtered.json"
OUTPUT_FILE = "cleaned_qna.json"

# Load dataset
with open(INPUT_FILE, "r") as f:
    data = json.load(f)
print(f"Total: {len(data)} questions. Filtering...")

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # You can tune this based on your VRAM

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def make_prompt(question):
    return (
        "[INST] "
        "Is the following question meaningful and contextually specific — not a generic or vague template question? "
        "Answer with YES or NO only.\n\n"
        f"Question: {question} [/INST]"
    )

def batched_filter(questions):
    prompts = [make_prompt(q) for q in questions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return ["YES" in decoded[r][len(prompts[r]):].upper() for r in range(len(decoded))]

filtered_data = []
for i in tqdm(range(0, len(data), BATCH_SIZE)):
    batch = data[i:i+BATCH_SIZE]
    questions = [item["question"] for item in batch]
    # print(questions)
    keep_mask = batched_filter(questions)
    for keep, item in zip(keep_mask, batch):
        if keep:
            filtered_data.append(item)

# Save
with open(OUTPUT_FILE, "w") as f:
    for item in filtered_data:
        f.write(json.dumps(item) + "\n")

print(f"✅ Done! Filtered QnAs saved to {OUTPUT_FILE} ({len(filtered_data)} kept)")