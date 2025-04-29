from fastapi import FastAPI
from pydantic import BaseModel
import os
import openai
from pinecone import Pinecone

# Initialize OpenAI and Pinecone
openai.api_key = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("am-academy-content-index")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        # Step 1: Embed the query using OpenAI (correct model!)
        response = openai.embeddings.create(
            input=request.query,
            model="text-embedding-3-large"   # <- IMPORTANT: match your index!
        )
        query_vector = response.data[0].embedding

        # Step 2: Query Pinecone
        query_result = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )

        # Step 3: Prepare a simple response
        results = []
        for match in query_result.matches:
            metadata = match.metadata
            if metadata:
                results.append(metadata.get('text', 'No text found'))

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}
