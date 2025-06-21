from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import openai
import base64
import os
import re
from difflib import SequenceMatcher
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

# Your full traits_to_regions dictionary here (unchanged for brevity)
traits_to_regions = {
    # [Your long trait-to-region dictionary remains unchanged]
}

def find_traits(text):
    matches = []
    for trait in traits_to_regions:
        similarity = SequenceMatcher(None, trait.lower(), text.lower()).ratio()
        if similarity > 0.6 or re.search(rf"\b{re.escape(trait)}\b", text, re.IGNORECASE):
            matches.append(trait)
    return matches

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
        raw_text = response.choices[0].message.content.strip()

        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        bullet_points = "\n".join(f"• {line}" for line in lines)

        matched_traits = find_traits(raw_text)
        matched_regions = []
        for trait in matched_traits:
            matched_regions.extend(traits_to_regions.get(trait, []))

        if matched_regions:
            region_counts = Counter(matched_regions)
            top_region, _ = region_counts.most_common(1)[0]
        else:
            # If fuzzy fails entirely, choose the most similar trait manually
            all_traits = list(traits_to_regions.keys())
            best_trait = max(all_traits, key=lambda t: SequenceMatcher(None, t.lower(), raw_text.lower()).ratio())
            top_region = traits_to_regions[best_trait][0]

        region_output = f"\n\n🌍 **Most Associated Region:** *{top_region}*\nBased on morphological trait clustering."
        full_result = f"{bullet_points}{region_output}"

        return {
            "result": full_result,
            "regions": [top_region]
        }

    except Exception as e:
        return {"error": str(e)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
