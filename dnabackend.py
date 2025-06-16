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
                "You are a facial phenotype analysis model. You observe human anatomical features and describe them "
                "using objective, scientific language. You avoid references to race, ethnicity, identity, or cultural origin."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please analyze the individual's visible facial morphology in precise and concise scientific terms. "
                        "Include descriptions of skin tone and undertone, craniofacial proportions, nasal bridge and width, cheekbones, "
                        "brow ridge, forehead slope, chin and jaw shape, eye shape and spacing, eyelid type, and hair color and texture. "
                        "Use compact anatomical descriptors (e.g., 'broad alar base', 'projected chin', 'low nasal root'). "
                        "Then briefly mention which broad morphological clusters (not specific populations) such a trait pattern statistically overlaps with, "
                        "without claiming origin or identity. Frame this as a morphological observation task only."
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
            temperature=0.6  # safe + exploratory
        )
        result = response.choices[0].message.content
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
