from openai import OpenAI
from PIL import Image
import base64
import os
import json
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI"))

IMAGE_DIR = "./images"

def image_to_base64(path, max_size=(1024, 1024)):
    img = Image.open(path)
    img.thumbnail(max_size)

    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=85)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

results = []

for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue

    path = os.path.join(IMAGE_DIR, filename)
    try:
        base64_image = image_to_base64(path)

        response = client.responses.create(
            model="gpt-5-nano",
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": """
                    Describe this image in 2-3 sentences.
                    Then give exactly 15 keywords.
                    Return JSON like:
                    {
                      "description": "...",
                      "keywords": ["...", "..."]
                    }
                    """
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }],
        )

        try:
            parsed = json.loads(response.output_text)
        except json.JSONDecodeError:
            parsed = response.output_text

        results.append({
            "filename": filename,
            "result": parsed
        })
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        results.append({
            "filename": filename,
            "error": str(e)
        })

with open("image_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done")