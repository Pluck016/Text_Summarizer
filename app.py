from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device("cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once during startup so web requests process immediately."""
    global model, tokenizer
    MODEL_NAME = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    yield

app = FastAPI(
    title="Text Summarizer App", 
    description="Text Summarization using T5", 
    version="1.0",
    lifespan=lifespan
)

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

def run_inference(dialogue: str) -> str:
    """CPU inference function executed off the main event loop."""
    cleaned_text = clean_data(dialogue)
    input_text = "summarize: " + cleaned_text

    inputs = tokenizer(
        input_text,
        max_length=256,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=80,
            min_length=15,
            num_beams=1,
            early_stopping=True
        )
    
    return tokenizer.decode(targets[0], skip_special_tokens=True)

# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    try:
        # Offload CPU heavy work to prevent blocking the web gateway
        summary = await run_in_threadpool(run_inference, dialogue_input.dialogue)
        return {"summary": summary}
    except Exception as e:
        return {"summary": f"Inference Error: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request, "result": ""}
    )