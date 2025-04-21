import requests
from pinecone import Pinecone, ServerlessSpec
import time
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize Pinecone with your API key
pc = Pinecone(api_key="pcsk_5Rdqat_Sx9R8WoQHqcK485RtbxDwFkA59VmhvNt3F3EiL9qeMfENo233Qp89fsRuh9p2Tu")

index_name = "unc-cs-index"

def ingest(data):
    """
    Data: List of dictionaries with 'id' and 'text' keys
    """
    embeddings = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[d['text'] for d in data],
        parameters={
            "input_type": "passage"
        }
    )

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    index = pc.Index(index_name)

    vectors = []
    for d, e in zip(data, embeddings):
        vectors.append({
            "id": d['id'],
            "values": e['values'],
            "metadata": {'text': d['text']}
        })

    index.upsert(
        vectors=vectors,
        namespace="ns1"
    )


# Assuming ingest and inference functions are implemented elsewhere
# You can import them here if they are in a separate file.
# from your_module import ingest, inference

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# 1. PUT /ingest_documents: Endpoint to ingest documents
@app.put("/ingest_documents")
def ingest_documents(data: List[Document]):
    try:
        # Convert the incoming data to the expected format for your ingest function
        documents = [{"doc_id": doc.doc_id, "text": doc.text} for doc in data]
        ingest(documents)  # Call your ingest function
        return {"message": "Documents ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")

# 2. GET /query: Endpoint to query the documents
@app.get("/query")
async def query_documents(query: str, top_k: int = 5) -> QueryResponse:
    try:
        async def inference(query: str, top_k: int):
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
        results = await inference(query, top_k)
        
        if not results:
            raise HTTPException(status_code=404, detail="No matching documents found")
        
        fin_resp = {"query": query, "top_k": top_k, "results": results['matches']}
        return fin_resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")
    

import json

# Replace with your Google Gemini API key
API_KEY = "AIzaSyANjsBsvuXnytos4V6HN86fYQ9A9a3sRiY"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

# Function to call Gemini 1.5 Pro API
def call_gemini(prompt):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        try:
            response_json = response.json()
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Error: Unexpected response format"
    else:
        return f"Error: {response.status_code}, {response.text}"

@app.get("/answer_question")
async def get_response(query: str, useRag: bool) -> Any:
    if useRag:
        try:
            # Convert the incoming data to the expected format for your ingest function
            sources = await query_documents(query, 5)
            prompt = "Answer the question honestly, but in a patient and explanatory manner, from your own knowledge and the sources mentioned below. If you do find the question irrelevant, output only NONE. Do not try to guess the answer.\n"
            for source in sources['results']:
                prompt += f"Source: {source['metadata']['text']}\n"
            prompt += f"\nQuestion: {query}\nAnswer:"
            response = call_gemini(prompt)
            return {"message": response, "sources": [{"doc_id": source['metadata']['id'], "text": source['metadata']['text']} for source in sources['results']]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error while inferencing documents: {str(e)}")
    else:
        try:
            prompt = query
            response = call_gemini(prompt)
            return {"message": response, "sources": []}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error while inferencing documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
