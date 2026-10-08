from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="LLM Deployment API")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "LLM API is running"}

@app.post("/generate")
def generate(request: PromptRequest):
    result = generator(
        request.prompt,
        max_new_tokens=50
    )

    return {"response": result[0]["generated_text"]}