from fastapi import FastAPI
from app.inference import load_model, generate

app = FastAPI(title="Word RNN Generator")

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/")
def home():
    return {"message": "Word RNN API Running"}

@app.get("/generate")
def generate_text(prompt: str, temperature: float = 1.0):
    text = generate(prompt, temperature=temperature)
    return {"generated_text": text}
