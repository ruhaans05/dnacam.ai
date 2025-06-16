from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import openai
import base64
import os

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_face(image: UploadFile = File(...)):
    contents = await image.read()
    base64_image = base64.b64encode(contents).decode("utf-8")
    mime_type = image.content_type or "image/jpeg"
    image_data = f"data:{mime_type};base64,{base64_image}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a morphological analysis assistant. You extract and describe physical traits using neutral anatomical language. "
                "You do not reference race, ethnicity, or identity. You work only with pattern-based trait data and regional morphological frequencies."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please analyze the individual's visible facial features using anatomical and scientific terms. "
                        "Describe traits such as skin tone, undertone, eye shape and spacing, nasal structure, craniofacial shape, jawline, and hair texture. "
                        "Then for each trait, list 1–2 regions where this trait is statistically common according to population-level morphological datasets. "
                        "At the end, identify which region appears most frequently across the trait mappings, and output that as the statistically most similar region. "
                        "Avoid any identity or cultural interpretation. This is a frequency-based anatomical comparison only."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data,
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=900,
            temperature=0.6
        )
        result = response.choices[0].message.content
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
