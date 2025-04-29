# main.py

import os
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone

import openai

app = FastAPI()

# Initialize Pinecone (new style)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Your Pinecone index
index = pc.Index("am-academy-content-index")

# Initialize OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(request: QueryRequest):
    # Search in Pinecone
    query_result = index.query(
        vector=[0.0]*1536,  # (placeholder - you should generate real embeddings)
        top_k=1,
        include_metadata=True
    )

    # Use the first match
    text = query_result['matches'][0]['metadata']['text']

    # Ask OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant for Additive Manufacturing."},
            {"role": "user", "content": f"Based on the following content, answer this question:\n\nContent: {text}\n\nQuestion: {request.query}"}
        ]
    )

    return {"answer": response['choices'][0]['message']['content']}
