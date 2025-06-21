from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import openai
import base64
import os
import re
from collections import Counter
import difflib

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

traits_to_regions = {
    "broad nasal base": ["Congo Basin", "Eastern India", "Philippines", "West Africa"],
    "deep melanin pigmentation": ["Congo Basin"],
    "medium brown skin": ["Eastern India", "Central India"],
    "thick wavy hair": ["Eastern India"],
    "flat nasal bridge": ["Philippines", "Vietnam"],
    "almond eyes": ["Philippines", "Southeast Asia", "Central Asia"],
    "olive skin": ["South Asia", "Pakistan"],
    "prominent nose bridge": ["South Asia"],
    "Mediterranean complexion": ["Southern Italy"],
    "rounded jawline": ["Central India", "Korea"],
    "Amazonian features": ["Northern Brazil"],
    "small orbital opening": ["Sri Lanka"],
    "epicanthic fold": ["Mongolia", "Korea", "Japan", "Siberia"],
    "round facial shape": ["Mongolia"],
    "bronzed skin": ["Andean Highlands", "Andes"],
    "narrow eyes": ["Andean Highlands"],
    "intermediate skin tone": ["Yunnan"],
    "high forehead": ["Northern Europe"],
    "light pigmentation": ["Northern Europe", "Scandinavia", "UK"],
    "hooked nose": ["Caucasus"],
    "straight eyebrows": ["Northern China", "Korea", "Kazakhstan"],
    "flat face": ["Northern China"],
    "medium pigmentation": ["Northern China", "France"],
    "broad cheekbones": ["Kazakhstan", "Northern Canada", "Siberia"],
    "projecting jawline": ["East Africa", "Melanesia"],
    "tight curls": ["Melanesia"],
    "low nasal bridge": ["Southeast Asia", "Tibet", "Borneo"],
    "wider facial angle": ["Tibet"],
    "broad nose": ["Borneo", "West Africa"],
    "narrow nasal bridge": ["Northern India", "Arabian Peninsula", "Iran Plateau", "Spain"],
    "longer face": ["Northern India"],
    "deep eyes": ["Arabian Peninsula"],
    "prominent chin": ["Iran Plateau"],
    "high cheekbones": ["Central Asia", "East Africa", "Andes", "Himalayas"],
    "broad forehead": ["Himalayas", "Ukraine"],
    "light eyes": ["Scandinavia"],
    "tall stature": ["Scandinavia"],
    "narrow face": ["Scandinavia"],
    "high melanin": ["West Africa"],
    "shorter forehead": ["West Africa"],
    "diverse blend": ["East Coast USA", "West Coast USA"],
    "mixed features": ["East Coast USA"],
    "multiethnic appearance": ["West Coast USA"],
    "European-African mix": ["Southern USA"],
    "wide facial width": ["Southern USA"],
    "Inuit cranial traits": ["Northern Canada"],
    "cold-adapted noses": ["Northern Canada"],
    "Mesoamerican features": ["Mexico"],
    "straight black hair": ["Mexico"],
    "African-Caribbean mix": ["Caribbean"],
    "medium-wide nose": ["Caribbean"],
    "Khoisan features": ["South Africa"],
    "Bantu cranial shape": ["South Africa"],
    "Berber-Arab features": ["Maghreb"],
    "medium skin tone": ["Maghreb", "France"],
    "Anatolian skull": ["Turkey"],
    "mixed traits": ["Turkey"],
    "Slavic-Dinaric nose": ["Balkans"],
    "robust jawline": ["Balkans"],
    "Anglo-Celtic features": ["Australia"],
    "oval face": ["Australia", "France"],
    "Māori brow ridge": ["New Zealand"],
    "strong jaw": ["New Zealand"],
    "short limbs": ["Siberia"],
    "brown skin": ["Indonesia"],
    "broad lips": ["Indonesia"],
    "straight hair": ["Japan"],
    "oval jaw": ["Japan"],
    "Anglo-Saxon features": ["UK"],
    "thin lips": ["UK"],
    "Gallic traits": ["France"],
    "Teutonic jaw": ["Germany"],
    "deep eye sockets": ["Germany"],
    "Mediterranean pigment": ["Spain"],
    "Slavic cranial vault": ["Ukraine"],
    "Indo-Iranian features": ["Pakistan"],
    "Bengali curve": ["Bangladesh"],
    "soft cheekbones": ["Bangladesh"],
    "Bamar structure": ["Myanmar"],
    "medium-dark skin": ["Myanmar"],
    "Thai eye ridge": ["Thailand"],
    "smaller jaw": ["Thailand"],
    "Malay jaw curve": ["Malaysia"],
    "medium nasal bridge": ["Malaysia"]
}

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
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        bullet_points = "\n".join(f"• {line}" for line in lines)

        matched_regions = []
        detected_traits = []

        for trait, regions in traits_to_regions.items():
            if re.search(rf"\b{re.escape(trait)}\b", raw_text, re.IGNORECASE):
                matched_regions.extend(regions)
                detected_traits.append(trait)

        # Fuzzy fallback if no traits matched directly
        if not matched_regions and not detected_traits:
            all_traits = list(traits_to_regions.keys())
            for line in raw_text.split('\n'):
                close_matches = difflib.get_close_matches(line.lower(), all_traits, n=1, cutoff=0.6)
                if close_matches:
                    fallback_trait = close_matches[0]
                    matched_regions.extend(traits_to_regions[fallback_trait])
                    detected_traits.append(fallback_trait)

        if matched_regions:
            region_counts = Counter(matched_regions)
            top_region, _ = region_counts.most_common(1)[0]
            region_output = f"\n\n🌍 **Most Associated Region:** *{top_region}*\nBased on morphological trait clustering."
        else:
            top_region = "South Asia"
            region_output = f"\n\n🌍 **Most Associated Region:** *{top_region}*\n"

        full_result = f"{bullet_points}{region_output}"

        return {
            "result": full_result,
            "regions": [top_region]
        }

    except Exception as e:
        return {"error": f"❌ OpenAI error: {str(e)}"}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
