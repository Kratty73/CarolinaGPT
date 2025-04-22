import json
import time
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from trl import DPOTrainer, DPOConfig
import torch
from accelerate import Accelerator

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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
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
