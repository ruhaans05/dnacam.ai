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
            "role": "user",
            "content": [
                {
    "type": "text",
    "text": (
        "Based on every visible detail in this image — including but not limited to skin tone, hair type and color, face structure, eye shape and color, nose and lip shape, and other phenotypic features — perform a detailed, unbiased analysis. "
        "Your goal is to determine the most likely subregional geographic origin of this person (for example, a specific district, province, or small ethnic group) based on their physical features. "
        "Avoid vague generalizations, avoid modern political boundaries, and rely only on known human phenotypic diversity data. "
        "This is for a comprehensive ancestry report, so go into depth. Be as precise, objective, and specific as possible in your reasoning."
    )
}
,
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data
                    }
                }
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500
        )
        result = response.choices[0].message.content
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# Mount static files AFTER defining routes
app.mount("/", StaticFiles(directory=".", html=True), name="static")
