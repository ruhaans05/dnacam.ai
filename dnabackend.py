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

    extract_traits_messages = [
        {
            "role": "system",
            "content": (
                "You are a morphological feature analysis model. Describe only anatomical features based on the image. "
                "Avoid identity, cultural, or political commentary."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "List the individual's visible traits using anatomical terms. "
                        "Include skin tone, eye shape, craniofacial proportions, nasal structure, and hair texture. "
                        "Use a neutral scientific tone."
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
        # Step 1: Extract traits from image
        traits_response = client.chat.completions.create(
            model="gpt-4o",
            messages=extract_traits_messages,
            max_tokens=700,
            temperature=0.5
        )
        traits_summary = traits_response.choices[0].message.content

        # Step 2: Feed those traits into a second GPT query to get the most likely region
        region_prompt = [
            {
                "role": "system",
                "content": (
                    "You are an anatomical data analyst. Given a list of human physical traits, identify the region(s) "
                    "where these traits are most statistically concentrated. Base your answer on trait frequency patterns, "
                    "not cultural or identity assumptions."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Here is a list of physical traits observed in a human face:\n\n{traits_summary}\n\n"
                    "Based on population trait clustering data, identify the single region of the world where this combination "
                    "of traits is most statistically concentrated. Do not mention ethnicity or culture."
                )
            }
        ]

        region_response = client.chat.completions.create(
            model="gpt-4o",
            messages=region_prompt,
            max_tokens=300,
            temperature=0.5
        )
        region_result = region_response.choices[0].message.content

        return {
            "traits": traits_summary,
            "region": region_result
        }

    except Exception as e:
        return {"error": str(e)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
