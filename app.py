import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

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


@app.post("/answer-image")
def answer(req: Request):

    prompt = f"""
Answer ONLY the question using the image.

Question:
{req.question}

Rules:
- Return ONLY the answer.
- If it is numeric return only the number.
- No currency symbols.
- No units.
- No explanation.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            types.Part.from_bytes(
                data=__import__("base64").b64decode(req.image_base64),
                mime_type="image/png",
            ),
        ],
    )

    return {
        "answer": response.text.strip()
    }


@app.get("/")
def root():
    return {"status": "running"}