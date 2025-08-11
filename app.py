from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class TextRequest(BaseModel):
    texts: list[str]

@app.post("/embed")
def embed(request: TextRequest):
    embeddings = model.encode(request.texts)
    return {"embeddings": embeddings.tolist()}
