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
                "Analyze the individual's visible anatomical traits with dense, scientific descriptors. Include: "
                "skin tone, undertone, craniofacial shape, forehead slope, cheekbone position, nasal width/bridge, "
                "lip thickness, jaw shape, chin prominence, eye shape and distance, eyelid type, brow ridge, and hair color/texture. "
                "Frame the analysis like a compressed trait vector used in morphological clustering (e.g., PCA or k-NN). "
                "At the end, output the most statistically similar subregional cluster (e.g., western Kenya, eastern Uttar Pradesh, "
                "rural Guangxi, northern Andes, coastal Maghreb, etc.) using only population-level phenotypic similarity — "
                "not culture, race, or identity."
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
