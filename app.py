import os
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

templates = Jinja2Templates(directory=".")

# Option A: Paste token directly here, or Option B: set HF_TOKEN in Render Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HUGGINGFACE_TOKEN_HERE")
API_URL = "https://api-inference.huggingface.co/models/t5-small"

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

class DialogueInput(BaseModel):
    dialogue: str

def summarize_via_hf_api(text: str) -> str:
    payload = {
        "inputs": f"summarize: {text}",
        "parameters": {"max_length": 150, "min_length": 30}
    }
    
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("summary_text", "No summary text generated.")
        return str(result)
    elif response.status_code == 503:
        return "Model is currently loading on Hugging Face. Please try again in 20 seconds!"
    else:
        return f"Hugging Face API Error ({response.status_code}): {response.text}"

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_via_hf_api(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request, "result": ""}
    )