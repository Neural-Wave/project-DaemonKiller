from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from rag import RAG
from generation import generate
import logging
logging.basicConfig(
    filename='app.log',          # Log file name
    level=logging.INFO,           # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

app = FastAPI()


# Define a Pydantic model for the incoming request
class QueryRequest(BaseModel):
    class Config:
        extra = 'allow'

# @app.post("/generate-response/")
def generate_response(request: QueryRequest):
    # Construct the prompt using the provided query and context
    try:
        rag = RAG(n_retrieved=10)
        context = rag.query(request.query)
        response = generate(query=request.query, context=context)
        response_json = response.json()

        # return {"status": "success", "response": response_json.get("response")}
        logging.info(f"Received request: {request.model_dump()}")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"External API request failed: {e}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Response from external API is not in JSON format.")
    

if __name__ == "__main__":
    generate_response()
