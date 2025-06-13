from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import json
from typing import List, Dict
import requests
from pinecone import Pinecone, ServerlessSpec
import time
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tqdm import tqdm
from peft import PeftModel

from transformers import StoppingCriteria

access_token = "#ACCESS TOKEN HERE"

# Load base model
base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
adapter_paths = ["./dpo-llama-output/checkpoint-104", "./dpo-llama-output/checkpoint-208", "./dpo-llama-output/checkpoint-309"]  # or latest checkpoint you have

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # automatically load to GPU
    token=access_token
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('./Llama-3.1-8B-Instruct', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, model, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            num_beams=4,
            length_penalty=3.0,
            early_stopping=True
            # stopping_criteria=MyStoppingCriteria("Question:", prompt)
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


with open('dpo_ready_data_test.json', 'r') as f:
    data = json.load(f)
output = []
# for adapter_path in tqdm(adapter_paths):
#     # Load LoRA adapters
#     model = PeftModel.from_pretrained(model, adapter_path)
for idx, q in tqdm(enumerate(data)):
    response = generate_response(q['prompt'], model)
    output.append({'model': 'llama_8b', 'prompt': q['prompt'], 'answer': response})

with open('llama_8b.json', 'w') as f:
    json.dump(output, f, indent=2)
# prompt = "Source: Biography. Huaxiu Yao is an assistant professor in the Department of Computer Science and the School of Data Science and Society at the University of North Carolina at Chapel Hill. Prior to joining UNC, he was a postdoctoral scholar in the IRIS Lab in the Department of Computer Science at Stanford University hosted by Professor Chelsea Finn. He received a doctorate in 2021 at Pennsylvania State University under the advisory of Professor Zhenhui \u201cJessie\u201d Li. During his Ph.D. study, he also spent time visiting the SAILING Lab at Carnegie Mellon University, hosted by Professor Eric P. Xing.\nSource: Contact. Biography Huaxiu Yao is an assistant professor in the Department of Computer Science and the School of Data Science and Society at the University of North Carolina at Chapel Hill. Prior to joining UNC, he was a postdoctoral scholar in the IRIS Lab in the Department of Computer Science at Stanford University hosted by Professor Chelsea Finn. He received a doctorate in 2021 at Pennsylvania State University under the advisory of Professor Zhenhui \u201cJessie\u201d Li. During his Ph.D. study, he also spent time visiting the SAILING Lab at Carnegie Mellon University, hosted by Professor Eric P. Xing.\nSource: . Contact 254 Sitterson Hall huaxiu@cs.unc.edu https://www.huaxiuyao.io/ Biography Huaxiu Yao is an assistant professor in the Department of Computer Science and the School of Data Science and Society at the University of North Carolina at Chapel Hill. Prior to joining UNC, he was a postdoctoral scholar in the IRIS Lab in the Department of Computer Science at Stanford University hosted by Professor Chelsea Finn. He received a doctorate in 2021 at Pennsylvania State University under the advisory of Professor Zhenhui \u201cJessie\u201d Li. During his Ph.D. study, he also spent time visiting the SAILING Lab at Carnegie Mellon University, hosted by\nSource: Associate Professor, Department of Psychiatry. (167) Ph.D. 2007, Shanghai Jiao Tong. Huaxiu Yao Assistant Professor 254 Sitterson Hall huaxiu@cs.unc.edu (183) Ph.D. 2021, Penn State.\nSource: Huaxiu Yao. (function($) { $(\"body, html\").addClass(\"heels_full_width_overflow\"); })(jQuery); Huaxiu Yao Assistant Professor (183) Ph.D. 2021, Penn State. (function($) { $(\"body, html\").addClass(\"heels_full_width_overflow\"); })(jQuery); Contact 254 Sitterson Hall huaxiu@cs.unc.edu https://www.huaxiuyao.io/ Biography Huaxiu Yao is an assistant professor in the Department of Computer Science and the School of Data Science and Society at the University of North Carolina at Chapel Hill. Prior to joining UNC, he was a postdoctoral scholar in the IRIS Lab in the Department of Computer Science at Stanford University hosted by Professor Chelsea Finn. He received a doctorate in 2021 at Pennsylvania State University under the advisory of Professor Zhenhui\nBased on the sample question answer format.\nIn which year was the computer science department established at UNC?\nThe UNC Department of Computer Science was established in 1964. Then answer only one question based solely of given text and answer 'No records found' if no relevant information is provided *and give no other output*. Fill the answer followed by a '\n' \\n In which year did Huaxiu Yao receive his doctorate?\nAnswer:"
# for adapter_path in tqdm(adapter_paths):
    # Load LoRA adapters
    # model = PeftModel.from_pretrained(model, adapter_path)
# print(generate_response(prompt, model))
