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

# JSON content for trait-to-region fallback
TRAIT_REGION_DATA = {
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

# Save to file if missing
TRAIT_FILE = "traits_to_regions.json"
if not os.path.exists(TRAIT_FILE):
    with open(TRAIT_FILE, "w") as f:
        json.dump(TRAIT_REGION_DATA, f, indent=2)

# Load trait-to-region mapping
with open(TRAIT_FILE) as f:
    traits_to_regions = json.load(f)

def extract_regions_from_text(text):
    matched_regions = []
    for trait, regions in traits_to_regions.items():
        if re.search(rf"\b{re.escape(trait)}\b", text, re.IGNORECASE):
            matched_regions.extend(regions if isinstance(regions, list) else [regions])
    if matched_regions:
        region_counts = Counter(matched_regions)
        top_regions = [region for region, _ in region_counts.most_common(3)]
        return top_regions
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
                        "Consider traits such as craniofacial proportions, skin pigmentation, eye morphology, and hair texture. "
                        "Describe how these traits may be similar to those observed in specific regional morphological clusters "
                        "based on population-level trait datasets, without making assumptions about identity, race, or origin. "
                        "Do not use nationality or cultural terms. Frame your reasoning as pattern-based classification, not sociological interpretation."
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
        result = response.choices[0].message.content
        regions = extract_regions_from_text(result)
        return {
            "result": result,
            "regions": regions
        }
    except Exception as e:
        return {"error": str(e)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
