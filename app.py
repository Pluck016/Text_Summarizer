from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

MODEL_NAME = "t5-small"
model = None
tokenizer = None
device = torch.device("cpu")

def get_model_and_tokenizer():
    """Load model into memory on-demand to minimize startup RAM overhead."""
    global model, tokenizer
    if model is None or tokenizer is None:
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
    return model, tokenizer

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue: str) -> str:
    m, t = get_model_and_tokenizer()
    cleaned_text = clean_data(dialogue)

    # Required T5 prompt prefix
    input_text = "summarize: " + cleaned_text

    inputs = t(
        input_text,
        max_length=256,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Disable gradient tracking & use greedy search (num_beams=1) to stay well under 512MB RAM
    with torch.no_grad():
        targets = m.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=80,
            min_length=15,
            num_beams=1,
            early_stopping=True
        )
    
    summary = t.decode(targets[0], skip_special_tokens=True)
    return summary

# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request, "result": ""}
    )