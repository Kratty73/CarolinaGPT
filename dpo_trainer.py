import json
import time
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from trl import DPOTrainer, DPOConfig
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Paths
MODEL_PATH = "./Llama-3.1-8B-Instruct"
DATA_PATH = "dpo_ready_data_train.json"

accelerator = Accelerator()
device = accelerator.device

def print_memory_usage(tag=""):
    if not accelerator.is_main_process:
        return
    print(f"\n>>> {tag}")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        print(f"GPU {i}: Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# Load cleaned QnA dataset
def load_qna_dataset(path):
    with open(path) as f:
        data = json.load(f)

    # Convert to DPO format
    rows = []
    for item in data:
        rows.append({
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        })

    return Dataset.from_list(rows)

# Load models and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
model.to(device)

# Add LoRA adapters
lora_config = LoraConfig(
    r=8,                        # Low-rank dimension
    lora_alpha=16,               # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers (adjust if needed based on model architecture)
    lora_dropout=0.05,           # Dropout for LoRA
    bias="none",                 # Bias setting
    task_type="CAUSAL_LM"        # For language models
)

model = get_peft_model(model, lora_config)

# Enable gradient checkpointing if you want memory savings (you already did)
model.gradient_checkpointing_enable()

print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
dataset = load_qna_dataset(DATA_PATH)

# Timer callback
class TimerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.epoch_start = time.time()
        print('EPOCH Start')

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed = time.time() - self.epoch_start
        print(f"Epoch {state.epoch} finished in {elapsed / 60:.2f} minutes")
    
    def on_step_end(self, args, state, control, **kwargs):
        print_memory_usage(tag=f"Step {state.global_step}")

class MemoryMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print_memory_usage(tag=f"Step {state.global_step}")

# DPO Config
training_args = DPOConfig(
    output_dir="dpo-llama-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=5e-6,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
    beta=0.1,
    truncation_mode="keep_start",
    generate_during_eval=False,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[TimerCallback(), MemoryMonitorCallback()]
)

trainer.train()
