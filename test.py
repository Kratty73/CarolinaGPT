from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pinecone import Pinecone, ServerlessSpec

# Set local model path
base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
pc = Pinecone(api_key="Your API key here")
access_token = "Your Access token here"

index_name = "unc-cs-index"

# Load tokenizer from local path
tokenizer = AutoTokenizer.from_pretrained('./Llama-3.1-8B-Instruct')
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # automatically load to GPU
    token=access_token
)

# Prompt for inference
print('Type in your query')
query = str(input())
print('Fetching doccuments...')
embedding = pc.inference.embed(
    model="llama-text-embed-v2",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)
index = pc.Index(index_name)
results = index.query(
    namespace="ns1",
    vector=embedding[0].values,
    top_k=5,
    include_values=False,
    include_metadata=True
)
print('Done')

def generate_response(prompt, model, max_new_tokens=200):
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

sources=results['matches']
prompt = ""
for source in sources:
    prompt += "Source: " + source['metadata']['text'] + '\n'
prompt += "Based on the sample question answer format.\nIn which year was the computer science department established at UNC?\nThe UNC Department of Computer Science was established in 1964. Now answer only one question based solely of given text and answer 'No records found' if no relevant information is provided *and give no other output*. Fill the answer followed by a '\n' \n"
prompt += "Question: " + query + '\n' + 'Answer:'
print('Generating Response...')
output = generate_response(prompt, model)
print('Done')

# Display output
print(output[len(prompt):].split('\n')[0])