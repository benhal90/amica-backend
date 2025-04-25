from fastapi import FastAPI
from pydantic import BaseModel
import os
import openai
import pinecone

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

class QueryRequest(BaseModel):
    query: str

@app.post("/pinecone_search")
async def pinecone_search(request: QueryRequest):
    embed = openai.Embedding.create(
        input=request.query,
        model="text-embedding-3-large"
    )["data"][0]["embedding"]

    results = index.query(vector=embed, top_k=3, include_metadata=True)
    texts = [match["metadata"]["text"] for match in results["matches"]]
    return {"results": "\n\n".join(texts)}
