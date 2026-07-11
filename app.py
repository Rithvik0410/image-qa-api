import os
import base64
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Request(BaseModel):
    image_base64: str
    question: str


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/answer-image")
def answer(req: Request):
    try:
        # Handle both raw base64 and data:image/...;base64,...
        image_b64 = req.image_base64.strip()

        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_b64)

        # Detect image type
        if image_bytes.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif image_bytes.startswith(b"\x89PNG"):
            mime = "image/png"
        else:
            mime = "application/octet-stream"

        prompt = f"""
Answer ONLY the user's question using the image.

Question:
{req.question}

Rules:
- Return ONLY the answer.
- The answer must be a string.
- If numeric, return only the number.
- No units.
- No currency symbols.
- No explanation.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime,
                ),
            ],
        )

        answer = ""

        if hasattr(response, "text") and response.text:
            answer = response.text.strip()
        else:
            try:
                answer = response.candidates[0].content.parts[0].text.strip()
            except Exception:
                answer = ""

        return {"answer": str(answer)}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
