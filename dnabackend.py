from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import openai
import base64
import os
import json
import re
from collections import Counter

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

# Embedded trait-to-region mapping
traits_to_regions = {
    "broad nasal base": ["Congo Basin", "Eastern India", "Philippines"],
    "wavy black hair": ["South Asia", "Southern Italy", "Vietnam"],
    "medium brown skin tone": ["Central India", "Northern Brazil", "Sri Lanka"],
    "prominent cheekbones": ["Mongolia", "Andean Highlands", "Yunnan"],
    "deep-set eyes": ["Northern Europe", "Caucasus"],
    "straight eyebrows": ["Northern China", "Korea", "Kazakhstan"],
    "projecting jawline": ["East Africa", "Melanesia"],
    "low nasal bridge": ["Southeast Asia", "Tibet", "Borneo"],
    "narrow nasal bridge": ["Northern India", "Arabian Peninsula", "Iran Plateau"],
    "high cheekbones": ["Central Asia", "Andes", "Himalayas"]
}

def extract_regions_from_text(text):
    matched_regions = []
    for trait, regions in traits_to_regions.items():
        if re.search(rf"\b{re.escape(trait)}\b", text, re.IGNORECASE):
            matched_regions.extend(regions)
    if matched_regions:
        region_counts = Counter(matched_regions)
        return [region for region, _ in region_counts.most_common(3)]
    return []

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
                "You're a morphological pattern analysis model. You describe visible human facial traits using anatomical terminology only. "
                "You do not mention identity, ethnicity, culture, or nationality. You strictly follow a scientific and neutral tone."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe the visible anatomical traits in the image using neutral, clinical language. "
                        "Focus on bone structure, facial proportions, eye shape, nasal structure, skin tone, and hair pattern. "
                        "Avoid any cultural or geographic references. Use only descriptive, morphological terminology without interpretation."
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
            temperature=0.7
        )
        raw_text = response.choices[0].message.content

        # Format traits as bullet points
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        bullet_points = "\n".join(f"• {line}" for line in lines)

        regions = extract_regions_from_text(raw_text)

        return {
            "result": bullet_points,
            "regions": regions
        }

    except Exception as e:
        return {"error": str(e)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
