from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import openai
import base64

# Set your OpenAI API key
openai.api_key = "sk-..."

app = FastAPI()

# Allow frontend connection (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_face(image: UploadFile = File(...)):
    # Read image file and convert to base64
    contents = await image.read()
    base64_image = base64.b64encode(contents).decode("utf-8")
    image_data = f"data:image/jpeg;base64,{base64_image}"

    # GPT-4 Vision prompt
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this person's facial features, complexion, and structure. "
                        "Then, based on global human variation and visible phenotype, guess which region or regions of the world "
                        "they most likely resemble ancestrally. Focus on phenotypic similarity, not modern nationality. "
                        "Be specific (country or subregion level) and explain your reasoning clearly."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": image_data
                }
            ],
        }
    ]

    # Call OpenAI Vision API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=500
        )
        result = response.choices[0].message["content"]
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
