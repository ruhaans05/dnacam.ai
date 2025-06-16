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
                "You are a morphological feature classification model. "
                "You use facial trait data to mathematically map individuals to regions using only anatomical data. "
                "You do not refer to culture, ethnicity, identity, or sociopolitical categories — only physical structure and scientific reasoning."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Simulate a K-Nearest Neighbors (k-NN) algorithm that compares visible traits "
                        "(skin tone, eye shape, nose structure, jawline, hair texture, craniofacial proportions, etc.) "
                        "to a dataset of global human morphological variation. "
                        "Output the most likely matching microregion in the world — e.g., eastern Uttar Pradesh, western Kenya, rural Guangxi, northern Peru. "
                        "Describe the anatomical traits neutrally and scientifically. "
                        "Do not reference race, identity, or nationality. "
                        "This is a mathematical pattern classification, not a sociological analysis."
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
            max_tokens=700,
            temperature=0.2
        )
        result = response.choices[0].message.content
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
