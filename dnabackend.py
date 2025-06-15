from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import openai
import base64
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve all frontend files from root
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# Serve index.html explicitly on root route
@app.get("/")
def serve_home():
    return FileResponse("index.html")


@app.post("/analyze")
async def analyze_face(image: UploadFile = File(...)):
    contents = await image.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

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
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ],
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # or your authorized model
            messages=messages,
            max_tokens=500
        )
        result = response.choices[0].message["content"]
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
