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

# Initialize Pinecone with your API key
pc = Pinecone(api_key="INSERT API KEY HERE")

index_name = "unc-cs-index"

# Pydantic model to validate input for ingest_documents
class Document(BaseModel):
    doc_id: str
    text: str

class Match(BaseModel):
    id: str
    metadata: Any
    score: float
    values: List[float]

class QueryResponse(BaseModel):
    query: str
    top_k: int
    results: List[Match]

class RAGResponse(BaseModel):
    message: str
    sources: List[Document]

def query_documents(query: str, top_k: int = 5) -> QueryResponse:
    try:
        def inference(query: str, top_k: int):
            """
            Sample Response: 
            {'matches': [{'id': 'vec3',
                        'metadata': {'text': 'Many people enjoy eating apples as a '
                                            'healthy snack.'},
                        'score': 0.025584612,
                        'values': []},
                      {'id': 'vec5',
                        'metadata': {'text': 'An apple a day keeps the doctor away, as '
                                            'the saying goes.'},
                        'score': 0.00986214262,
                        'values': []},
                      {'id': 'vec4',
                        'metadata': {'text': 'Apple Inc. has revolutionized the tech '
                                            'industry with its sleek designs and '
                                            'user-friendly interfaces.'},
                        'score': -0.00467887754,
                        'values': []}],
            'namespace': 'ns1',
            'usage': {'read_units': 6}}
            """
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
                top_k=top_k,
                include_values=False,
                include_metadata=True
            )

            return results

        # Call your inference function
        results = inference(query, top_k)
        
        if not results:
            raise HTTPException(status_code=404, detail="No matching documents found")
        
        fin_resp = {"query": query, "top_k": top_k, "results": results['matches']}
        return fin_resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")
    


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

with open('dpo_ready_data_test.json', 'r') as f:
    data = json.load(f)

output = []

full_sources = []
for q in tqdm(data):
    full_sources.append(query_documents(q["prompt"]))

for t in tqdm([0.0, 0.3, 0.7, 1.0, 1.3]):
    for idx, q in tqdm(enumerate(data)):
        sources = full_sources[idx]
        prompt = "Answer the question honestly, but in a patient and explanatory manner, from your own knowledge and the sources mentioned below. If you do find the question irrelevant, output only NONE. Do not try to guess the answer.\n"
        for source in sources['results']:
            prompt += f"Source: {source['metadata']['text']}\n"
        prompt += f"\nQuestion: {q['prompt']}\nAnswer:"
        response = generator(prompt, max_new_tokens=50, do_sample=t>0, temperature=t, repetition_penalty=1.2)[0]["generated_text"][len(prompt):].strip()
        output.append({'temp': t, 'prompt': q['prompt'], 'answer': response})

with open('llama_raw_output.json', 'w') as f:
    json.dump(output, f, indent=2)