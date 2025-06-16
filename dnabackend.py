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
                "You are a morphological feature analysis model. "
                "You describe human anatomical traits using objective scientific terminology. "
                "You do not make cultural, racial, or identity-based assumptions."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please analyze this face using concise anatomical terminology. "
                        "List detailed features such as: skin tone and undertone, brow ridge structure, forehead slope, nose bridge height and width, nasal tip angle, philtrum depth, lip thickness, jaw width, gonial angle, chin shape, eye shape and distance, eyelid type, hair color and texture. "
                        "Describe each feature briefly but precisely, using dense language (e.g., 'broad alar base with recessed nasion', 'olive undertone with intermediate melanin'). "
                        "Then, based on pattern alignment with population-level morphological data, name the single most statistically similar regional cluster of origin (e.g., southeast Odisha, highland Eritrea, rural Guangxi, eastern Uttar Pradesh, central Guatemala). "
                        "Avoid identity terms, do not mention ethnicity or nationality. This is a scientific pattern-based classification only."
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
            max_tokens=750,
            temperature=0.2
        )
        result = response.choices[0].message.content
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
