from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
    image_data = f"data:image/jpeg;base64,{base64_image}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an objective phenotype analysis model. "
                "You do not make identity guesses or personal claims. "
                "You only describe phenotypic features visible in the image and compare them to known human diversity data. "
                "The goal is to support scientific understanding of visible traits, not to assign ethnicity or origin."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please analyze this image based on physical features like skin tone, hair texture and color, facial structure, eye shape and color, and any other visible morphology. "
                        "Describe which global population clusters (e.g., tropical African, North Indian, Andean, etc.) these traits statistically align with in physical anthropology studies. "
                        "Avoid generalizations, political terms, or identity-based assumptions. Just describe the patterns scientifically."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=600,
            temperature=0.3  # makes it more factual
        )
        result = response.choices[0].message.content
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# Mount static files AFTER routes
app.mount("/", StaticFiles(directory=".", html=True), name="static")
