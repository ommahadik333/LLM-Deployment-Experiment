
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(title="LLM Deployment API")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

class PromptRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=1000)

@app.get("/")
def home():
    return {"message": "LLM API is running"}

@app.post("/generate")
def generate(request: PromptRequest):
    try:
        result = generator(
            "Answer the following question clearly "
            "in 2 to 4 complete sentences: "
            + request.prompt,
            max_new_tokens=100,
            num_beams=4,
            do_sample=False,
            repetition_penalty=1.2,
            truncation=True
        )

        answer = result[0]["generated_text"].strip()
        return {"response": answer}

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Text generation failed"
        )
