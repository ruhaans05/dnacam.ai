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
                "You are a morphological feature analysis model. You only analyze anatomical traits and describe patterns "
                "observed in the image using population-level morphological data. You avoid cultural, political, or identity-based assumptions."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Use a scientific and anatomical approach to describe the visible facial features in the image. "
                        "List traits such as craniofacial proportions, skin pigmentation, nasal shape, eye structure, jawline, and hair texture using compact scientific language. "
                        "For each listed trait, specify 1–2 regions where that trait is statistically common, based on morphological population datasets. "
                        "At the end, determine the single region that appears most frequently across all the traits. "
                        "Output that as the most statistically similar region. Do not mention ethnicity, culture, or origin. "
                        "This is a feature frequency–based anatomical clustering task only."
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
            temperature=0.7
        )
        result = response.choices[0].message.content
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
